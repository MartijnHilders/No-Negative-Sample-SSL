import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import random


class WassOrderDistance(nn.Module):
    """
        Compute the Order-Preserving Wasserstein Distance (OPW) for two sequences X and Y.

        Parameters:
        - X: (N x d) tensor of N d-dimensional vectors
        - Y: (M x d) tensor of M d-dimensional vectors
        - lamda1: weight of IDM regularization, default value: 50
        - lamda2: weight of KL-divergence regularization, default value: 0.1
        - delta: parameter of prior Gaussian distribution, default value: 1
        - verbose: whether to display iteration status, default value: 0 (not display)

        Returns:
        - dis: OPW distance between X and Y
        - T: learned transport between X and Y, a (N x M) tensor

        -------------
        c : barycenter according to weights
        ADVICE: divide M by median(M) to have a natural scale
        for lambda

        -------------
        Copyright (c) 2017 Bing Su, Gang Hua
        -------------

        -------------
        License
        The code can be used for research purposes only.
        """

    def __init__(self, lamda1=0.1, lamda2=0.01, delta=10, max_iter=200, verbose=0):
        super(WassOrderDistance, self).__init__()
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.delta = delta
        self.max_iter = max_iter
        self.verbose = verbose


    # todo try to change so that it works with 3d -> batch, num_frames,
    # output -> batch size [batches, 2] where (2 = distance, transport plan)
    # save batch (keep this version and create new version [1,
    def forward(self, X, Y):
        tolerance = 0.5e-2

        # set the p_norm
        p_norm = np.inf

        n = X.shape[1]
        m = Y.shape[1]
        dim = X.shape[2]
        if Y.shape[2] != dim:
            print("\nThe dimensions of instances in the input sequences must be the same!")
            return

        # fill p tensor
        i = torch.arange(1, n + 1, dtype=torch.float64, device=X.device).view(-1, 1)
        j = torch.arange(1, m + 1, dtype=torch.float64, device=X.device).view(1, -1)
        mid_para = torch.sqrt(torch.tensor(1 / (n ** 2) + 1 / (m ** 2)))
        d = (i / n - j / m).abs() / mid_para
        p = torch.exp(-d ** 2 / (2 * self.delta ** 2)) / (self.delta * torch.sqrt(2 * torch.tensor(np.pi)))

        #fill s tensor
        s = torch.zeros((n, m), dtype=torch.float64, device=X.device)
        i = torch.arange(1, n + 1, dtype=torch.float64, device=X.device).view(-1, 1)
        j = torch.arange(1, m + 1, dtype=torch.float64, device=X.device).view(1, -1)
        s = self.lamda1 / ((i / n - j / m) ** 2 + 1)

        # normalize and get the pairwise distances between X and Y (square euclidean)
        X_norm = (X - X.mean(dim=1, keepdim=True) / torch.sqrt(X.var(dim=1, keepdim=True) + 0.0001))
        Y_norm = (Y - Y.mean(dim=1, keepdim=True) / torch.sqrt(Y.var(dim=1, keepdim=True) + 0.0001))

        d = pdist2_EucSq(X_norm, Y_norm)

        # scale down the distance matrix by max of every batch and calculate k-matrix
        d = d / d.flatten(start_dim=-2).max(dim=1)[0].view(-1, 1, 1)# first flatten to easily capture max of whole matrix
        k = p * torch.exp((s - d)/self.lamda2)

        # k[k<torch.exp(-100)] = torch.exp(torch.tensor(-100))

        # With some parameters, some entries of K may exceed the maching-precision
        # limit; in such cases, you may need to adjust the parameters, and/or
        # normalize the input features in sequences or the matrix D; Please see the
        # paper for details.
        # In practical situations it might be a good idea to do the following:
        # K[K<1e-100] = 1e-100

        a = (torch.ones((n, 1), dtype=torch.float64, device=X.device)/n)
        b = (torch.ones((m, 1), dtype=torch.float64, device=X.device)/m)

        ainvK = torch.divide(k, a)
        compt = 0
        u = (torch.ones((n, 1), dtype=torch.float64, device=X.device)/n)

        # The Sinkhorn's fixed point iteration
        # This part of code is adopted from the code "sinkhornTransport.m" by Marco
        # Cuturi; website: http://marcocuturi.net/SI.html
        # Relevant paper:
        # M. Cuturi,
        # Sinkhorn Distances : Lightspeed Computation of Optimal Transport,
        # Advances in Neural Information Processing Systems (NIPS) 26, 2013
        while compt < self.max_iter:
            u = 1/(torch.matmul(ainvK, (b/(torch.matmul(k.mT, u)))))
            compt += 1

            if compt % 20 == 1 or compt == self.max_iter:
                v = b/(torch.matmul(k.mT, u))
                u = 1/(torch.matmul(ainvK, v))

                criterion = torch.norm(torch.sum(torch.abs(v * (torch.matmul(k.mT, u)) - b)), p=p_norm)
                if criterion < tolerance or torch.isnan(criterion):
                    break

                compt += 1
                if self.verbose > 0:
                    print(f"Iteration: {compt} Criterion: {criterion}")


        U = torch.multiply(k, d)
        dis = torch.sum(u * (torch.matmul(U, v)), dim=1, keepdim=True)
        t = torch.multiply(v.mT, torch.multiply(u, k))

        return dis, t


# calculate pairwise distance, with the squared euclidean distance as distance metric.
# todo reference https://github.com/pdollar/toolbox/blob/master/classify/pdist2.m
def pdist2_EucSq(x, y):
    x_square = torch.sum(x * x, dim=2, keepdim=True)
    y_square = torch.sum(y.mT * y.mT, dim=1, keepdim=True)
    d = x_square + y_square - 2 * torch.matmul(x, y.mT)
    return d


def test_1():
    # two equal arrays, so perfect matching
    x = np.array(
        [[1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4],
         [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2],
         [3, 4], [5, 6]])

    y = np.array(
       [[1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4],
         [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2],
         [5, 6], [3, 4]])

    x = np.tile(x, (128, 1, 1))
    y = np.tile(y, (128, 1, 1))

    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    x.requires_grad = True
    y.requires_grad = True

    order_dist = WassOrderDistance()
    distance, t_scheme = order_dist(x, y)


    for i in range(5):

        idx = random.randint(0, t_scheme.shape[0])
        print(f'distance: {distance[idx]}')
        create_trans_heatmap(t_scheme[idx])

    # check if backprop works
    distance[0].backward()
    print(distance)


# testing two different batches
def test_2():
    x_1 = np.array(
        [[1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4],
         [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2],
         [3, 4], [5, 6], [1, 2], [1,2], [1,2]])

    y_1 = np.array(
        [[1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4],
         [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2],
         [3, 4], [5, 6], [1, 2], [1, 2],[1,2]])

    x_2 = np.array(
        [[1, 2], [3, 4], [5, 6], [4, 3], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4],
         [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2],
         [3, 4], [5, 6], [1, 2], [8,99]])

    y_2 = np.array(
        [[1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4],
         [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2],
         [3, 4], [5, 6], [1, 2], [4, 3], [8,99]])


    x = np.stack([x_1, x_2])
    y = np.stack([y_1, y_2])

    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    x.requires_grad = True
    y.requires_grad = True

    order_dist = WassOrderDistance()
    distance, t_scheme = order_dist(x, y)

    for i in range(2):
        print(f'distance: {distance[i]}')
        create_trans_heatmap(t_scheme[i])

    # check if backprop works
    distance[0].backward()
    print(distance)



def create_trans_heatmap(trans):
    sns.heatmap(trans.detach().numpy())
    plt.show()


if __name__ == '__main__':
    test_2()

import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np


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

    def __init__(self, lamda1=50, lamda2=0.1, delta=1, max_iter=20, verbose=0):
        super(WassOrderDistance, self).__init__()
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.delta = delta
        self.max_iter = max_iter
        self.verbose = verbose


    def forward(self, X, Y):
        tolerance = 0.5e-2

        # set the p_norm and convert the input tensors to float32 to reduce memory usage
        p_norm = np.inf

        n = X.shape[0]
        m = Y.shape[0]
        dim = X.shape[1]
        if Y.shape[1] != dim:
            print("The dimensions of instances in the input sequences must be the same!")
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
        X_norm = (X - X.mean(dim=0)/torch.sqrt(X.var(dim=0) + 0.0001))
        Y_norm = (Y - Y.mean(dim=0)/torch.sqrt(Y.var(dim=0) + 0.0001))
        d = pdist2_EucSq(X_norm, Y_norm)

        k = p * torch.exp((s - d)/self.lamda2)

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
            u = 1/(ainvK @ (b/(k.T @ u)))
            compt += 1
            if compt % 20 == 1 or compt == self.max_iter:
                v = b/(k.T @ u)
                u = 1/(ainvK @ v)

                criterion = torch.norm(torch.sum(torch.abs(v * (k.T @ u) - b)), p=p_norm)
                if criterion < tolerance or torch.isnan(criterion):
                    break

                compt += 1
                if self.verbose > 0:
                    print(f"Iteration: {compt} Criterion: {criterion}")


        U = torch.multiply(k, d)
        dis = torch.sum(u * (U @ v))
        t = torch.multiply(v.T, torch.multiply(u, k))

        return dis, t


# calculate pairwise distance, with the squared euclidean distance as distance metric.
# todo reference https://github.com/pdollar/toolbox/blob/master/classify/pdist2.m
def pdist2_EucSq(x, y):
    x_square = torch.sum(x * x, dim=1, keepdim=True)
    y_square = torch.sum(y.T * y.T, dim=0, keepdim=True)
    d = x_square + y_square - 2 * torch.mm(x, y.T)
    return d


def test():
    # two equal arrays, so perfect matching
    x = np.array(
        [[1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4],
         [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2],
         [3, 4], [5, 6]])

    y = np.array(
        [[1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4],
         [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2],
         [5, 6], [3, 4]])

    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    x.requires_grad = True
    y.requires_grad = True

    order_dist = WassOrderDistance()
    distance, t_scheme = order_dist(x, y)
    print(f'distance: {distance}')


    create_trans_heatmap(t_scheme)

    # check if backprop works
    distance.backward()
    print(distance)


def create_trans_heatmap(trans):
    sns.heatmap(trans.detach().numpy())
    plt.show()


if __name__ == '__main__':
    test()

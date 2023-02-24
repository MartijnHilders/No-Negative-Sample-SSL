import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import random
from matplotlib.colors import LogNorm

#todo reference
# " code taken from VolTA implementation of GOT

class WassOrderDistanceGromov(nn.Module):

    def __init__(self, lamda1=0.1, lamda2=0.01, delta=10, max_iter=200, verbose=0):
        super(WassOrderDistanceGromov, self).__init__()
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.delta = delta
        self.max_iter = max_iter
        self.verbose = verbose

    def cost_matrix_batch_torch(self, x, y):
        "Returns the cosine distance batchwise"
        # x is the image feature: bs * d * m * m
        # y is the audio feature: bs * d * nF
        # return: bs * n * m
        # print(x.size())
        bs = list(x.size())[0]
        D = x.size(1)
        assert (x.size(1) == y.size(1))
        x = x.contiguous().view(bs, D, -1)  # bs * d * m^2
        x = x.div(torch.norm(x, p=2, dim=1, keepdim=True) + 1e-12)
        y = y.div(torch.norm(y, p=2, dim=1, keepdim=True) + 1e-12)
        cos_dis = torch.bmm(torch.transpose(x, 1, 2), y)  # .transpose(1,2)
        cos_dis = 1 - cos_dis  # to minimize this value
        # cos_dis = - cos_dis
        return cos_dis.transpose(2, 1)


    def IPOT_torch_batch_uniform(self, C, bs, n, m, beta=0.5, iteration=50):
        # C is the distance matrix
        # c: bs by n by m
        sigma = torch.ones(bs, int(m), 1).cuda() / float(m)
        T = torch.ones(bs, n, m).cuda()
        A = torch.exp(-C / beta).float().cuda()
        for t in range(iteration):
            Q = A * T  # bs * n * m
            for k in range(1):
                delta = 1 / (n * torch.bmm(Q, sigma))
                a = torch.bmm(torch.transpose(Q, 1, 2), delta)
                sigma = 1 / (float(m) * a)
            T = delta * Q * sigma.transpose(2, 1)

        return T  # .detach()

    def IPOT_distance_torch_batch_uniform(self, C, bs, n, m, iteration=50):
        C = C.float().cuda()
        T = self.IPOT_torch_batch_uniform(C, bs, n, m, iteration=iteration)
        temp = torch.bmm(torch.transpose(C, 1, 2), T)
        distance = self.batch_trace(temp, m, bs)
        return distance

    def cos_batch_torch(self, x, y):
        "Returns the cosine distance batchwise"
        # x is the image feature: bs * d * m * m
        # y is the audio feature: bs * d * nF
        # return: bs * n * m
        # print(x.size())
        bs = x.size(0)
        D = x.size(1)
        assert (x.size(1) == y.size(1))
        x = x.contiguous().view(bs, D, -1)  # bs * d * m^2
        x = x.div(torch.norm(x, p=2, dim=1, keepdim=True) + 1e-12)
        y = y.div(torch.norm(y, p=2, dim=1, keepdim=True) + 1e-12)
        cos_dis = torch.bmm(torch.transpose(x, 1, 2), y)  # .transpose(1,2)
        cos_dis = 1 - cos_dis  # to minimize this value
        # return cos_dis.transpose(2,1)
        # TODO:
        beta = 0.1
        min_score = cos_dis.min()
        max_score = cos_dis.max()
        threshold = min_score + beta * (max_score - min_score)
        res = cos_dis - threshold
        # res = torch.nn.ReLU()

        return torch.nn.functional.relu(res.transpose(2, 1))

    def GW_distance(self, X, Y, p, q, lamda=0.5, iteration=5, OT_iteration=20):
        '''
        :param X, Y: Source and target embeddings , batchsize by embed_dim by n
        :param p, q: probability vectors
        :param lamda: regularization
        :return: GW distance
        '''
        Cs = self.cos_batch_torch(X, X).float().cuda()
        Ct = self.cos_batch_torch(Y, Y).float().cuda()
        # pdb.set_trace()
        bs = Cs.size(0)
        m = Ct.size(2)
        n = Cs.size(2)
        T, Cst = self.GW_torch_batch(Cs, Ct, bs, n, m, p, q, beta=lamda, iteration=iteration, OT_iteration=OT_iteration)
        temp = torch.bmm(torch.transpose(Cst, 1, 2), T)
        distance = self.batch_trace(temp, m, bs)
        return distance

    def GW_torch_batch(self, Cs, Ct, bs, n, m, p, q, beta=0.5, iteration=5, OT_iteration=20):
        one_m = torch.ones(bs, m, 1).float().cuda()
        one_n = torch.ones(bs, n, 1).float().cuda()

        Cst = torch.bmm(torch.bmm(Cs ** 2, p), torch.transpose(one_m, 1, 2)) + \
              torch.bmm(one_n, torch.bmm(torch.transpose(q, 1, 2), torch.transpose(Ct ** 2, 1, 2)))  # bs by n by m
        gamma = torch.bmm(p, q.transpose(2, 1))  # outer product, init
        # gamma = torch.einsum('bi,bj->bij', (torch.squeeze(p), torch.squeeze(q))) # outer product, initialization
        for i in range(iteration):
            C_gamma = Cst - 2 * torch.bmm(torch.bmm(Cs, gamma), torch.transpose(Ct, 1, 2))
            # # Sinkhorn iteration
            # b = torch.ones(bs, m, 1).cuda()
            # K = torch.exp(-C_gamma/beta)
            # for i in range(50):cd
            # 	a = p/(torch.bmm(K, b))
            # 	b = q/torch.bmm(K.transpose(1,2), a)
            # gamma = a * K * b
            gamma = self.IPOT_torch_batch_uniform(C_gamma, bs, n, m, beta=beta, iteration=OT_iteration)
        Cgamma = Cst - 2 * torch.bmm(torch.bmm(Cs, gamma), torch.transpose(Ct, 1, 2))
        return gamma.detach(), Cgamma

    def GW_distance_uniform(self, X, Y, lamda=1e-1, iteration=5, OT_iteration=20):
        m = X.size(2)
        n = Y.size(2)
        bs = X.size(0)
        p = (torch.ones(bs, m, 1) / m).cuda()
        q = (torch.ones(bs, n, 1) / n).cuda()
        return self.GW_distance(X, Y, p, q, lamda=lamda, iteration=iteration, OT_iteration=OT_iteration)

    def batch_trace(self, input_matrix, n, bs):
        a = torch.eye(n).cuda().unsqueeze(0).repeat(bs, 1, 1)
        b = a * input_matrix
        return torch.sum(torch.sum(b, -1), -1).unsqueeze(1)

    # todo try to change so that it works with 3d -> batch, num_frames,
    # output -> batch size [batches, 2] where (2 = distance, transport plan)
    # save batch (keep this version and create new version [1,

    def forward(self, X, Y):

        # # Uniform implementation gromov
        # m = X.size(2)
        # n = Y.size(2)
        # bs = X.size(0)
        # p = (torch.ones(bs, m, 1) / m).cuda()
        # q = (torch.ones(bs, n, 1) / n).cuda()
        #
        # dis, t = self.GW_distance(X, Y, p, q, lamda=1e-1, iteration=5, OT_iteration=20)

        cos_distance = self.cost_matrix_batch_torch(X.transpose(2, 1), Y.transpose(2, 1))
        cos_distance = cos_distance.transpose(1, 2)
        beta = 0.1
        min_score = cos_distance.min()
        max_score = cos_distance.max()
        threshold = min_score + beta * (max_score - min_score)
        cos_dist = torch.nn.functional.relu(cos_distance - threshold)

        wd = self.IPOT_distance_torch_batch_uniform(cos_dist, X.size(0), X.size(1), Y.size(1), 30)
        gwd = self.GW_distance_uniform(X.transpose(2, 1), Y.transpose(2, 1))
        dis = torch.mean(gwd) + torch.mean(wd)
        return dis


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

    order_dist = WassOrderDistanceGromov()
    distance = order_dist(x, y)


    # for i in range(5):
    #
    #     idx = random.randint(0, t_scheme.shape[0])
    #     print(f'distance: {distance[idx]}')
    #     create_trans_heatmap(t_scheme[idx])

    # check if backprop works
    distance[0].backward()
    print(distance)


# testing two different batches
def test_2():
    x_1 = np.array(
        [[1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4],
         [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2],
         [3, 4], [5, 6], [1, 2], [1,2], [1,2], [1, 2], [3, 4], [5, 6], [1,2]])

    y_1 = np.array(
        [[1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4],
         [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2],
         [3, 4], [5, 6], [1, 2], [1, 2],[1,2], [1, 2], [3, 4], [5, 6], [1,2]])

    x_2 = np.array(
        [[1, 2], [3, 4], [5, 6], [4, 3], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4],
         [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2],
         [3, 4], [5, 6], [1, 2], [8,90], [1, 2], [3, 4], [5, 6], [93,2]])

    y_2 = np.array(
        [[1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4],
         [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2],
         [3, 4], [5, 6], [1, 2], [4, 3], [8,90], [93,2], [1, 2], [3, 4], [5, 6]])


    x = np.stack([x_1, x_2])
    y = np.stack([y_1, y_2])

    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    x.requires_grad = True
    y.requires_grad = True

    order_dist = WassOrderDistanceGromov()
    distance = order_dist(x, y)

    # for i in range(2):
    #     print(f'distance: {distance[i]}')
    #     create_trans_heatmap(t_scheme[i])

    # check if backprop works
    distance.backward()
    print(distance)



def create_trans_heatmap(trans):
    sns.heatmap(trans.cpu().detach().numpy())
    plt.show()


if __name__ == '__main__':
    test_2()

import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Adapted from https://github.com/gpeyre/SinkhornAutoDiff
class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, eps=0.1, max_iter=100, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x_in, y_in):
        # The Sinkhorn algorithm takes as input three variables :
        x = (x_in - x_in.mean(dim=1, keepdim=True) / torch.sqrt(x_in.var(dim=1, keepdim=True) + 0.0001))
        y = (y_in - y_in.mean(dim=1, keepdim=True) / torch.sqrt(y_in.var(dim=1, keepdim=True) + 0.0001))

        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]


        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False,  device=x.device).fill_(1.0 / x_points).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False, device=x.device).fill_(1.0 / y_points).squeeze()

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu + 1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau= -.8 ):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1


def test():
    x_1 = np.array(
        [[1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4],
         [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2],
         [3, 4], [5, 6], [1, 2], [1, 2], [1, 2]])

    y_1 = np.array(
        [[1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4],
         [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2],
         [3, 4], [5, 6], [1, 2], [1, 2], [1, 2]])

    x_2 = np.array(
        [[1, 2], [3, 4], [5, 6], [4, 3], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2],
         [3, 4],
         [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2],
         [3, 4], [5, 6], [1, 2], [8, 90]])

    y_2 = np.array(
        [[1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4],
         [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2],
         [3, 4], [5, 6], [1, 2], [4, 3], [8, 90]])


    x = np.stack([x_1, x_2])
    y = np.stack([y_1, y_2])

    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    x.requires_grad = True
    y.requires_grad = True

    order_dist = SinkhornDistance(eps=0.1, max_iter=100)
    distance, t_scheme, C = order_dist(x, y)

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
    test()

import numpy as np
import torch


def main():

    # two equal arrays, so perfect matching
    x = np.array([[1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6],[1, 2], [3, 4], [5, 6],[1, 2], [3, 4], [5, 6],[1, 2], [3, 4], [5, 6],[1, 2], [3, 4], [5, 6],[1, 2], [3, 4], [5, 6],[1, 2], [3, 4], [5, 6],[1, 2], [3, 4], [5, 6],[1, 2], [3, 4], [5, 6]])
    y = np.array(
        [[1, 2], [3, 4], [3, 4], [5, 6], [1, 2], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4],
         [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2],
         [3, 4], [5, 6]])

    # x = torch.from_numpy(x)
    # y = torch.from_numpy(y)


    distance, t_scheme = opw(x, y, verbose=0)
    print(f'distance: {distance}')

    #shuffle the array and check how the alignments are
    np.random.shuffle(y)
    distance, t_scheme = opw(x, y, verbose=0)
    print(f'distance: {distance}')



    # test the alignments
    # miss = 0
    # for idx in range(t_scheme.shape[0]):
    #     idx2 = np.argmax(t_scheme[idx])
    #
    #     if not np.array_equal(x[idx], y[idx2]):
    #         miss += 1
    #
    # print(miss)
    # err = (x.shape[0] - miss)/x.shape[0]
    # print(err)



#todo reference
def opw(X, Y, lamda1=50, lamda2=0.1, delta=1, verbose=0):
    """
    Compute the Order-Preserving Wasserstein Distance (OPW) for two sequences X and Y.

    Parameters:
    - X: (N x d) matrix of N d-dimensional vectors
    - Y: (M x d) matrix of M d-dimensional vectors
    - lamda1: weight of IDM regularization, default value: 50
    - lamda2: weight of KL-divergence regularization, default value: 0.1
    - delta: parameter of prior Gaussian distribution, default value: 1
    - verbose: whether to display iteration status, default value: 0 (not display)

    Returns:
    - dis: OPW distance between X and Y
    - T: learned transport between X and Y, a (N x M) matrix

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
    tolerance = 0.5e-2
    max_iter = 20

    # The maximum number of iterations; with a default small value, the
    # tolerance and VERBOSE may not be used;
    # Set it to a large value (e.g, 1000 or 10000) to obtain a more precise
    # transport;
    p_norm = np.inf

    n = X.shape[0]
    m = Y.shape[0]
    dim = X.shape[1]
    if Y.shape[1] != dim:
        print("The dimensions of instances in the input sequences must be the same!")
        return

    # fill p matrix
    p = np.zeros((n, m))
    mid_para = np.sqrt((1/(n**2) + 1/(m**2)))
    for i in range(1, n+1):
        for j in range(1, m+1):
            d = abs(i/n - j/m) / mid_para
            p[i-1, j-1] = np.exp(-d**2/(2*delta**2))/(delta*np.sqrt(2*np.pi))

    #fill s matrix
    s = np.zeros((n, m))
    for i in range(1, n+1):
        for j in range(1, m+1):
            s[i-1, j-1] = lamda1/((i/n - j/m)**2 + 1)

    # get the pairwise distances between X and Y
    d = pdist2_EucSq(X, Y)

    # check if we need to normalize
    if np.mean(d) > 1:
        d = d / np.max(d)

    # In cases the instances in sequences are not normalized and/or are very
    # high-dimensional, the matrix D can be normalized or scaled as follows:
    # D = D/max(max(D))
    # D = D/(10**2)

    k = p * np.exp((s - d)/lamda2)
    # With some parameters, some entries of K may exceed the maching-precision
    # limit; in such cases, you may need to adjust the parameters, and/or
    # normalize the input features in sequences or the matrix D; Please see the
    # paper for details.
    # In practical situations it might be a good idea to do the following:
    # K[K<1e-100] = 1e-100

    a = np.ones((n, 1))/n
    b = np.ones((m, 1))/m

    ainvK = np.divide(k, a)
    compt = 0
    u = np.ones((n, 1))/n

    # The Sinkhorn's fixed point iteration
    # This part of code is adopted from the code "sinkhornTransport.m" by Marco
    # Cuturi; website: http://marcocuturi.net/SI.html
    # Relevant paper:
    # M. Cuturi,
    # Sinkhorn Distances : Lightspeed Computation of Optimal Transport,
    # Advances in Neural Information Processing Systems (NIPS) 26, 2013
    while compt < max_iter:
        u = 1/(ainvK @ (b/(k.T @ u)))
        compt += 1
        if compt % 20 == 1 or compt == max_iter:
            v = b/(k.T @ u)
            u = 1/(ainvK @ v)

            Criterion = np.linalg.norm([np.sum(np.abs(v * (k.T @ u) - b))], p_norm)
            if Criterion < tolerance or np.isnan(Criterion):
                break

            compt += 1
            if verbose > 0:
                print(f"Iteration: {compt} Criterion: {Criterion}")

    U = k * d
    dis = np.sum(u * (U @ v))
    t = np.multiply(v.T, np.multiply(u, k))

    return dis, t


# calculate pairwise distance, with the squared euclidean distance as distance metric.
# todo reference https://github.com/pdollar/toolbox/blob/master/classify/pdist2.m
def pdist2_EucSq(x, y):
    xx = np.sum(x * x, axis=1)
    yy = np.sum(y.T * y.T, axis=0)
    d = xx[:, np.newaxis] + yy[np.newaxis, :] - 2 * np.dot(x, y.T)

    return d

if __name__ == '__main__':
    main()

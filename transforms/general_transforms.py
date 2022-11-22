import numpy as np
import torch

class Normalize:
    """
    Normalizes based on mean and std. Used by skeleton and inertial modalities
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / self.std

class ToTensor:
    def __call__(self, x):
        return torch.from_numpy(x)

class Permute:
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, x):
        return x.permute(self.shape)

class ToFloat:
    def __call__(self, x):
        return x.float()

# class Transpose:
#     def __init__(self, shape):
#         self.shape = shape
#
#     def __call__(self, x):
#         return np.transpose(x, self.shape)

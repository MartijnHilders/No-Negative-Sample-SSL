import numpy as np
import random


class DepthSampler:
    """
    Resamples a video from any size of timesteps to the given size
    """

    def __init__(self, size):
        """
        Initiate sampler with the size to resample to
        :param size: int
        """
        self.size = size

    def __call__(self, x):
        # print(x.shape)

        if x.shape[0] < self.size:
            raise ValueError(f'Sampler size must be smaller than the original size of the array; Original: {x.shape[0]},'
                             f' Sampler: {self.size}, {x.shape}')
        return self.down_sample(x)

    # down_sample the given array to the specified size
    def down_sample(self, x):

        # create chunks and randomly sample a frame from the chunk
        chunks = np.array_split(x, self.size)

        d_frames = None
        for chunk in chunks:
            frame = random.choice(chunk)

            if d_frames is None:
                d_frames = frame

            else:
                d_frames = np.dstack((d_frames, frame))

        d_frames = np.moveaxis(d_frames, -1, 0)

        return d_frames

# Todo create a cropper for this and inertial. maybe put them together
class DepthCropper:
    def __init__(self, size):
        """
        Initiate sampler with the size to resample to
        :param size: int
        """
        self.size = size

    def __call__(self, x):
        print(x.shape)

        if x.shape[0] < self.size:
            raise ValueError(
                f'Sampler size must be smaller than the original size of the array; Original: {x.shape[0]},'
                f' Sampler: {self.size}, {x.shape}')
        return self.down_sample(x)



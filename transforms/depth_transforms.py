import numpy as np
import random
from PIL import Image


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
            frame = random.choice(chunk)[np.newaxis, :]

            if d_frames is None:
                d_frames = frame

            else:
                d_frames = np.concatenate((d_frames, frame))

        return d_frames

# Todo create a cropper for this and inertial. maybe put them together
# class DepthCropper:

# reshape the height and width of the images
class DepthResize:

    def __init__(self, height, width):
        """
        Initiate sampler with the size to reshape to
        :param height: int
        :param width: int
        """
        self.height = height
        self.width = width

    def __call__(self, x):
        # check for RGB
        if np.ndim(x) > 3:
            x = self.resize_RGB(x)

        else:
            x = self.resize_GREY(x)

        return x

    # maybe fixate interpolation mode to NEAREST to also use when upsampling
    def resize_RGB(self, x):
        resized = None
        for idx in range(x.shape[0]):
            im = Image.fromarray(x[idx], mode="RGB").resize((self.width, self.height))
            ar = np.asarray(im)[np.newaxis]

            if resized is None:
                resized = ar

            resized = np.concatenate([resized, ar])

        return resized

    def resize_GREY(self, x):
        resized = None
        for idx in range(x.shape[0]):
            im = Image.fromarray(x[idx], mode="L").resize((self.width, self.height))
            ar = np.asarray(im)

            if resized is None:
                resized = ar

            resized = np.dstack([resized, ar])

        resized = np.transpose(resized, [2, 0, 1])
        return resized


class ToRGB:

    def __init__(self):
       pass

    def __call__(self, x):
        x = np.stack((x,) * 3, axis=-1)  # correct shape: {frames, height, width, channels}
        return x








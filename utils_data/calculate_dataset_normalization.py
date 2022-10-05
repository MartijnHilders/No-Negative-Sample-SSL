import argparse
from transforms.inertial_transforms import ToTensor
import torch
from torch.utils import data
from torch.utils.data.dataloader import DataLoader
import numpy as np
from PIL.JpegImagePlugin import JpegImageFile
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from tqdm import tqdm

from data_modules.constants import DATASET_PROPERTIES

def calculate_inertial(dataset: Dataset):
    """
    Calculates the mean and standard deviation for inertial data, for each channel, across the given dataset.
    Assumes the data is in the shape of (L x C), where L is the length of the sequence and C is the number of channels.
    """

    # Auxiliary tensors for sum and sum of squares across each channel.
    num_channels = dataset[0]['inertial'].shape[1]
    count = 0
    psum = torch.zeros(num_channels, dtype=torch.float64)
    psum_sq = torch.zeros(num_channels, dtype=torch.float64)

    # Loop through each sequence and compute sum and sum of squares.
    for data in tqdm(dataset):
        inertial_data = data['inertial']
        count += inertial_data.shape[0]
        psum += inertial_data.sum(axis = 0)
        psum_sq += (inertial_data ** 2).sum(axis = 0)

    # Compute final values.
    total_mean = psum / count
    total_var = (psum_sq / count) - (total_mean ** 2)
    total_std  = torch.sqrt(total_var)

    return total_mean, total_std

def calculate_image(dataset: Dataset, modality: str):
    """
    Calculates the mean and standard deviation for image data, for each channel, across the given dataset.
    Assumes the data is in the shape of (L x C x W x H), where L is the length of the sequence, W and H
    are width and height, and C is the number of channels.
    """

    # Auxiliary tensors for sum and sum of squares across each channel.
    num_channels = dataset[0][modality].shape[1]
    count = 0
    psum = torch.zeros(num_channels, dtype=torch.float64)
    psum_sq = torch.zeros(num_channels, dtype=torch.float64)

    # Loop through each sequence and compute sum and sum of squares.
    for data in tqdm(dataset):
        image_data = torch.from_numpy(data[modality]).float()
        count += image_data.shape[0] * image_data.shape[2] * image_data.shape[3] # number of pixels in entire sequence
        psum += image_data.sum(axis = [0, 2, 3])
        psum_sq += (image_data ** 2).sum(axis = [0, 2, 3])

    # Compute final values.
    total_mean = psum / count
    total_var = (psum_sq / count) - (total_mean ** 2)
    total_std  = torch.sqrt(total_var)

    return total_mean, total_std

if __name__ == '__main__':

    # Parse and validate input arguments.
    available_datasets = DATASET_PROPERTIES.keys()
    available_modalities = {}
    for dataset in DATASET_PROPERTIES:
        modalities = DATASET_PROPERTIES[dataset].dataset_class._supported_modalities(
        )
        available_modalities[dataset] = modalities

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=available_datasets, required=True)
    parser.add_argument('--modality', required=True,
                        help="Available modalities: %s" % available_modalities)
    args = parser.parse_args()

    dataset = args.dataset
    modality = args.modality
    if modality not in available_modalities[dataset]:
        print("Invalid modality '%s' for dataset '%s'! Available modalities: %s" % (
            modality, dataset, available_modalities[dataset]))
        exit(1)

    # Depending on the specified modality, compute means and standard deviations.
    data_module = DATASET_PROPERTIES[dataset].datamodule_class(modalities=[modality])
    data_module.setup()
    if modality == "inertial":
        total_mean, total_std = calculate_inertial(data_module._create_train_dataset())
    elif modality == "depth":
        # Lambda transform used to get the data in (L x C x W x H) shape.
        reshape_transform = torchvision.transforms.Lambda(lambda seq: np.expand_dims(np.transpose(seq, (2, 0, 1)), 1))
        data_module.train_transforms = {"depth": reshape_transform}
        total_mean, total_std = calculate_image(data_module._create_train_dataset(), "depth")
    elif modality == "rgb":
        # Lambda transform used to get the data in (L x C x W x H) shape.
        reshape_transform = torchvision.transforms.Lambda(lambda seq: np.transpose(seq, (0, 3, 1, 2)))
        data_module.train_transforms = {"rgb": reshape_transform}
        total_mean, total_std = calculate_image(data_module._create_train_dataset(), "rgb")

    print("Mean: %s" % total_mean)
    print("Stddev: %s" % total_std)

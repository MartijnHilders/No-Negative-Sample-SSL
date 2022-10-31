from typing import List
from numpy import int16
from torchvision import transforms
import os

import datasets.utd_mhad as utd_mhad
from data_modules.mmhar_data_module import MMHarDataset, MMHarDataModule
from transforms.inertial_transforms import InertialSampler
from transforms.inertial_augmentations import Jittering
from transforms.skeleton_transforms import SkeletonSampler
from transforms.general_transforms import ToTensor, ToFloat
from utils.experiment_utils import load_yaml_to_dict
import multiprocessing as mp
from time import time

UTD_DEFAULT_SPLIT = {
    "train": {"subject": [1, 3, 5]},
    "val": {"subject": [7]},
    "test": {"subject": [2, 4, 6, 8]}
}

class UTDDataset(MMHarDataset):
    @staticmethod
    def _supported_modalities() -> List[str]:
        return ["inertial", "skeleton", "depth"]
    
    @staticmethod
    def _get_data_for_instance(modality, path):
        if modality == "inertial":
            return utd_mhad.UTDInertialInstance(path).signal
        elif modality == "skeleton":
            return utd_mhad.UTDSkeletonInstance(path).joints
        elif modality == "depth":
            return utd_mhad.UTDDepthInstance(path).image

class UTDDataModule(MMHarDataModule):

    def __init__(self,
            # path: str = os.path.join(os.path.dirname(os.path.abspath(os.curdir)),'multimodal_har_datasets/utd_mhad'),
            path: str = "/home/data/multimodal_har_datasets/utd_mhad",
            modalities: List[str] = ["inertial", "skeleton"],
            batch_size: int = 32,
            split = UTD_DEFAULT_SPLIT,
            train_transforms = {},
            test_transforms = {},
			ssl = False,
			n_views = 2,
            num_workers = 6,
			limited_k = None):
        super().__init__(path, modalities, batch_size, split, train_transforms, test_transforms, ssl, n_views, num_workers, limited_k)

    def _create_dataset_manager(self) -> utd_mhad.UTDDatasetManager:
        return utd_mhad.UTDDatasetManager(self.path)

    def _create_train_dataset(self) -> MMHarDataset:
        return UTDDataset(self.modalities, self.dataset_manager, self.split["train"], transforms=self.train_transforms, ssl=self.ssl, n_views=self.n_views, limited_k=self.limited_k)

    def _create_val_dataset(self) -> MMHarDataset:
        return UTDDataset(self.modalities, self.dataset_manager, self.split["val"], transforms=self.test_transforms, ssl=self.ssl, n_views=self.n_views)

    def _create_test_dataset(self) -> MMHarDataset:
        return UTDDataset(self.modalities, self.dataset_manager, self.split["test"], transforms=self.test_transforms)

def try_num_workers():
    train_transforms = {
        "inertial": transforms.Compose([InertialSampler(150)]),
        "skeleton": SkeletonSampler(150)
    }

    for num_workers in range(2, mp.cpu_count(), 2):
        data_module = UTDDataModule(batch_size=64, train_transforms=train_transforms, num_workers=num_workers)
        data_module.setup()
        dl = data_module.train_dataloader()

        start = time()
        epoch = 0
        for b in dl:
            epoch += 1
            if epoch > 5:
                break
        end = time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))


if __name__ == '__main__':
    train_transforms = {
        "inertial": transforms.Compose([ToTensor(), ToFloat(), Jittering(0.05), InertialSampler(150)]),
        "skeleton": SkeletonSampler(100)
    }

    # try_num_workers()

    data_module = UTDDataModule(batch_size=8, train_transforms=train_transforms)
    data_module.setup()

    dl = data_module.train_dataloader()
    for b in dl:
        print(b.keys())
        print(b['label'].shape)
        print(b['inertial'].shape)
        print(b['skeleton'].shape)
        break

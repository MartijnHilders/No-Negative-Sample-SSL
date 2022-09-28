from typing import List
from numpy import int16
import os
from torchvision import transforms

import datasets.czu_mhad as czu_mhad
from data_modules.mmhar_data_module import MMHarDataset, MMHarDataModule
from transforms.inertial_transforms import InertialSampler
from transforms.inertial_augmentations import Jittering
from transforms.skeleton_transforms import SkeletonSampler
from transforms.general_transforms import ToTensor, ToFloat
from utils.experiment_utils import load_yaml_to_dict

#TODO WE NEED TO CONTINUE ADAPTING THE DATA MODULE

CZU_DEFAULT_SPLIT = {
    "train": {"subject": [1, 2]},
    "val": {"subject": [3]},
    "test": {"subject": [4, 5]}
}


class CZUDataset(MMHarDataset):
    @staticmethod
    def _supported_modalities() -> List[str]:
        return ["inertial", "skeleton", "depth"]

    @staticmethod
    def _get_data_for_instance(modality, path):
        if modality == "inertial":
            return czu_mhad.CZUInertialInstance(path).signal
        elif modality == "skeleton":
            return czu_mhad.CZUSkeletonInstance(path).joints
        elif modality == "depth":
            return czu_mhad.CZUDepthInstance(path).image


class CZUDataModule(MMHarDataModule):

    def __init__(self,
                 path: str = os.path.join(os.path.dirname(os.path.abspath(os.curdir)),'multimodal_har_datasets\czu_mhad'),
                 modalities: List[str] = ["depth", "skeleton", "inertial"], # changed from ["intertial", "skeleton"]
                 batch_size: int = 32,
                 split=CZU_DEFAULT_SPLIT,
                 train_transforms={},
                 test_transforms={},
                 ssl=False,
                 n_views=2,
                 num_workers=1,
                 limited_k=None):
        super().__init__(path, modalities, batch_size, split, train_transforms, test_transforms, ssl, n_views,
                         num_workers, limited_k)

    def _create_dataset_manager(self) -> czu_mhad.CZUDatasetManager:
        return czu_mhad.CZUDatasetManager(self.path)

    def _create_train_dataset(self) -> MMHarDataset:
        return CZUDataset(self.modalities, self.dataset_manager, self.split["train"], transforms=self.train_transforms,
                          ssl=self.ssl, n_views=self.n_views, limited_k=self.limited_k)

    def _create_val_dataset(self) -> MMHarDataset:
        return CZUDataset(self.modalities, self.dataset_manager, self.split["val"], transforms=self.test_transforms,
                          ssl=self.ssl, n_views=self.n_views)

    def _create_test_dataset(self) -> MMHarDataset:
        return CZUDataset(self.modalities, self.dataset_manager, self.split["test"], transforms=self.test_transforms)


if __name__ == '__main__':
    train_transforms = {
        "inertial": transforms.Compose([ToTensor(), ToFloat(), Jittering(0.05), InertialSampler(150)]),
        "skeleton": SkeletonSampler(100)
    }

    data_module = CZUDataModule(batch_size=8, train_transforms=train_transforms)
    data_module.setup()

    dl = data_module.train_dataloader()

    # todo mistake here with the data shape, make sure that all of the data is in the correct format
    # todo ask bulat for couple of files from utdMHAD, especially skeleton see problem skeleton_transforms.py

    for b in dl:
        print(b.keys())
        print(b['label'].shape)
        print(b['inertial'].shape)
        print(b['skeleton'].shape)
        break

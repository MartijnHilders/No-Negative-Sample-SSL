from typing import List
from torchvision import transforms

import datasets.mmact as mmact
from data_modules.mmhar_data_module import MMHarDataset, MMHarDataModule
from transforms.inertial_transforms import InertialSampler
from transforms.inertial_augmentations import Jittering
from transforms.skeleton_transforms import SkeletonSampler
from transforms.general_transforms import ToTensor, ToFloat
from transforms.depth_transforms import DepthSampler, DepthResize
import multiprocessing as mp
import time

MMACT_DEFAULT_SPLIT = {
    "train": {"subject": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]},
    "val": {"subject": [13, 14, 15]},
    "test": {"subject": [17, 18, 19, 20]}
}

class MMActDataset(MMHarDataset):
    @staticmethod
    def _supported_modalities() -> List[str]:
        return ["inertial", "skeleton", "rgb"]
    
    @staticmethod
    def _get_data_for_instance(modality, path):
        if modality == "inertial":
            return mmact.MMActInertialInstance(path).signal
        elif modality == "skeleton":
            return mmact.MMActSkeletonInstance(path).joints
        elif modality == 'rgb':
            return mmact.MMActRGBInstance(path).video

class MMActDataModule(MMHarDataModule):
    def __init__(self, 
            path: str = "/home/data/multimodal_har_datasets/mmact_new",
            modalities: List[str] = ["inertial", "skeleton", "rgb"],
            batch_size: int = 32,
            split = MMACT_DEFAULT_SPLIT,
            train_transforms = {},
            test_transforms = {},
			ssl = False,
			n_views = 2,
            num_workers = 6,
			limited_k = None):
        super().__init__(path, modalities, batch_size, split, train_transforms, test_transforms, ssl, n_views, num_workers, limited_k)

    def _create_dataset_manager(self) -> mmact.MMActDatasetManager:
        return mmact.MMActDatasetManager(self.path)

    def _create_train_dataset(self) -> MMHarDataset:
        return MMActDataset(self.modalities, self.dataset_manager, self.split["train"], transforms=self.train_transforms, ssl=self.ssl, n_views=self.n_views, limited_k=self.limited_k)

    def _create_val_dataset(self) -> MMHarDataset:
        return MMActDataset(self.modalities, self.dataset_manager, self.split["val"], transforms=self.test_transforms, ssl=self.ssl, n_views=self.n_views)

    def _create_test_dataset(self) -> MMHarDataset:
        return MMActDataset(self.modalities, self.dataset_manager, self.split["test"], transforms=self.test_transforms)


def try_num_workers():
    train_transforms = {
        "inertial": transforms.Compose([InertialSampler(150)]),
        "skeleton": SkeletonSampler(150)
    }

    for num_workers in range(2, mp.cpu_count(), 2):
        data_module = MMActDataModule(batch_size=64, train_transforms=train_transforms, num_workers=num_workers)
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
        "skeleton": SkeletonSampler(150),
        "rgb": transforms.Compose([DepthSampler(40), DepthResize(False, 90, 70)])
    }

    # try_num_workers()

    data_module = MMActDataModule(batch_size=8, train_transforms=train_transforms, num_workers=8)
    data_module.setup()

    dl = data_module.train_dataloader()

    for b in dl:
        print(b.keys())
        print(b['label'].shape)
        print(b['inertial'].shape)
        print(b['rgb'].shape)
        print(b['skeleton'].shape)
        break
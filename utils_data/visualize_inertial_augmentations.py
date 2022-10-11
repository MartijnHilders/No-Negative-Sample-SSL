import argparse
import json

from transforms.augmentation_utils import compose_random_augmentations
from plotter import plot_inertial_subplots
from data_modules.utd_mhad_data_module import UTDDataModule

def main(args):
    data_module = UTDDataModule(path='/home/data/multimodal_har_datasets/utd_mhad/', modalities=["inertial"], batch_size=16)
    data_module.setup()
    dataset = data_module.train_dataloader().dataset

    # Retrieve one sample.
    sample = dataset[args.dataset_idx]["inertial"]

    # Apply random augmentations.
    with open('./configs/augmentations.yaml') as json_file:
        transform_config = json.load(json_file)
    augmentation = compose_random_augmentations("inertial", transform_config)
    augmented_sample = augmentation(sample)

    # Plot comparison.
    plot_inertial_subplots(top_data=sample, bottom_data=augmented_sample,
                           top_title="Original", bottom_title="Augmented",
                           y_label="deg/sec", save=args.save, show_figure=args.show)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_idx', type=int, default=0, help='idx to plot from train dataset')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--no_show', dest='show', action='store_false')
    parser.set_defaults(save=False, show=True, compare=False)
    args = parser.parse_args()
    main(args)

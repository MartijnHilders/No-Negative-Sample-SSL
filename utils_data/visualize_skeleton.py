import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torchvision.transforms

from configs.skeleton_properties import SKELETON_PROPERTIES
from data_modules.constants import DATASET_PROPERTIES
from transforms.skeleton_transforms import *
from transforms.general_transforms import ToTensor, Permute, ToFloat

def main(args):
    transforms = {
        "skeleton": torchvision.transforms.Compose([
            SkeletonSampler(50),
            RecenterJoints(SKELETON_PROPERTIES[args.dataset]["center_joint"]),
            NormalizeDistances(*SKELETON_PROPERTIES[args.dataset]["normalization_joints"]),
            ToTensor(),
            Permute([1,2,0]),
            ToFloat(),
            # RandomRotation(3, 30, 35, 1, SKELETON_PROPERTIES[args.dataset]["center_joint"]),
            # RandomScale(3, 2, 2, SKELETON_PROPERTIES[args.dataset]["center_joint"]),
            # RandomShear(3, 0.1)
        ])
    }
    datamodule = DATASET_PROPERTIES[args.dataset].datamodule_class(modalities=["skeleton"], train_transforms=transforms)
    datamodule.setup()
    dataset = datamodule.train_dataloader().dataset

    # Retrieve one sample
    sample = dataset[args.dataset_idx]["skeleton"]
    print("Instance details:\n", dataset.data_tables["skeleton"].iloc[args.dataset_idx])

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.view_init(21, -51)

    # Create cubic bounding box to simulate equal aspect ratio
    X = sample[0, :, :]
    Y = sample[2, :, :]
    Z = sample[1, :, :]
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # set limits to display properly, or if dataset is normalized use just 0 and 1
    x_lim_min = mid_x - max_range
    x_lim_max = mid_x + max_range
    y_lim_min = mid_y - max_range
    y_lim_max = mid_y + max_range
    z_lim_min = mid_z - max_range
    z_lim_max = mid_z + max_range

    def draw_joints(frame):
        # Reset axes
        ax.clear()

        # Re-set limits to keep axis from moving
        ax.set_xlabel('X')
        ax.set_xlim(x_lim_min, x_lim_max)
        ax.set_ylabel('Z')
        ax.set_ylim(y_lim_min, y_lim_max)
        ax.set_zlabel('Y')
        ax.set_zlim(z_lim_min, z_lim_max)

        # Print joints as points
        ax.scatter(sample[0, frame, :], sample[2, frame, :], sample[1, frame, :])

        # Print lines connecting the joints
        for bone in SKELETON_PROPERTIES[args.dataset]["bones"]:
            ax.plot(
                [sample[0, frame, bone[0]], sample[0, frame, bone[1]]],
                [sample[2, frame, bone[0]], sample[2, frame, bone[1]]],
                [sample[1, frame, bone[0]], sample[1, frame, bone[1]]],
            )

    if args.continuous or args.frame is None:
        anim = animation.FuncAnimation(fig, func=draw_joints, frames=sample.shape[1], interval=100)
    else:
        draw_joints(args.frame)

    if args.save and not args.continuous:
        plt.savefig('skeleton_%s' % args.frame)

    if args.show:
        plt.show()


if __name__ == '__main__':

    available_datasets = DATASET_PROPERTIES.keys()
    available_datasets = list(filter(lambda d: "skeleton" in DATASET_PROPERTIES[d].dataset_class._supported_modalities(), available_datasets))

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=available_datasets, required=True)
    parser.add_argument('--frame', type=int, default=None, help='Frame to plot')
    parser.add_argument('--dataset_idx', type=int, default=0, help='idx to plot from train dataset')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--continuous', action='store_true')
    parser.add_argument('--no_show', dest='show', action='store_false')
    parser.set_defaults(save=False, continuous=False, normalize=False, show=True)
    args = parser.parse_args()
    main(args)

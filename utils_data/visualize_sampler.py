import argparse
import numpy as np
import os

from transforms.inertial_transforms import InertialSampler
from plotter import plot_inertial, plot_inertial_gyroscope_multiple
from data_modules.utd_mhad_data_module import UTDDataModule
from data_modules.czu_mhad_data_module import CZUDataModule

def main(args):
    # path = '/home/data/multimodal_har_datasets/utd_mhad'
    path = os.path.join(os.path.dirname(os.path.abspath(os.curdir)),'multimodal_har_datasets/czu_mhad')
    data_module = CZUDataModule(path = path, modalities=["inertial"], batch_size=16)
    data_module.setup()
    dataset = data_module.train_dataloader().dataset

    # Retrieve one sample
    sample = dataset[args.dataset_idx]["inertial"]
    print('Original step size: %d' % sample.shape[0])
    if args.sampler_size:
        plot_data = InertialSampler(args.sampler_size)(sample)
        append_to_title = ' - Sampler %d' % args.sampler_size
        print('New step size: %d' % plot_data.shape[0])
    else:
        plot_data = sample
        append_to_title = ''

    if args.compare:
        data = np.array([sample[:, 0], plot_data[:, 0]])
        plot_inertial_gyroscope_multiple(title='Gyroscope', y_label='deg/sec', data=data, save=args.save, show_figure=args.show, legends=['original', 'sampled'])
    else:
        plot_inertial(plot_data[:, :3], title='Gyroscope' + append_to_title, y_label='deg/sec', save=args.save,
                      show_figure=args.show)
        plot_inertial(plot_data[:, 3:], title='Accelerometer' + append_to_title, y_label='m/sec^2', save=args.save,
                      show_figure=args.show)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_idx', type=int, default=0, help='idx to plot from train dataset')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--sampler_size', type=int, default=None)
    parser.add_argument('--no_show', dest='show', action='store_false')
    parser.add_argument('--compare', action='store_true')
    parser.set_defaults(save=False, show=True, compare=True, sampler_size = 1500)
    args = parser.parse_args()
    main(args)

import argparse
import numpy as np

from transforms.inertial_augmentations import Jittering
from plotter import plot_inertial, plot_inertial_gyroscope_multiple
from data_modules.utd_mhad_data_module import UTDDataModule

def main(args):
    data_module = UTDDataModule(path='/home/data/multimodal_har_datasets/utd_mhad', modalities=["inertial"], batch_size=16)
    data_module.setup()
    dataset = data_module.train_dataloader().dataset

    # Retrieve one sample
    sample = dataset[args.dataset_idx]["inertial"]
    if args.jitter_factor:
        plot_data = Jittering(args.jitter_factor)(sample)
        append_to_title = ' - Jittering %.2f' % args.jitter_factor
    else:
        plot_data = sample
        append_to_title = ' - Original'

    if args.compare:
        data = np.array([sample[:, 0], plot_data[:, 0]])
        legends = ['Original', 'Jittered %.2f' % args.jitter_factor]
        plot_inertial_gyroscope_multiple(title='Gyroscope', y_label='deg/sec', legends=legends, data=data,
                                         save=args.save,
                                         show_figure=args.show)
    else:
        plot_inertial(plot_data, title='Gyroscope' + append_to_title, y_label='deg/sec', save=args.save,
                      show_figure=args.show)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_idx', type=int, default=0, help='idx to plot from train dataset')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--jitter_factor', type=float, default=None)
    parser.add_argument('--no_show', dest='show', action='store_false')
    parser.add_argument('--compare', action='store_true')
    parser.set_defaults(save=False, show=True, compare=False)
    args = parser.parse_args()
    main(args)

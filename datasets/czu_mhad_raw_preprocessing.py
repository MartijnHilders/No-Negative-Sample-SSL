import argparse
import os


class CZUraw():
    """
    pre-processes CZU-MHAD it into the UTD-MHAD format in the same folder.
    Expects the following files and folders under data_path:
        depth_mat
        sensor_mat
        skeleton_mat
    """

    def __init__(self, data_path) -> None:
        self.data_path = data_path

    def process_dataset(self):
        print('Processing depth data...')
        self.process_data('Depth')
        print('Processing inertial data...')
        self.process_data('Inertial')
        print('Processing skeleton data...')
        self.process_data('Skeleton')


    def process_data(self, type):
        if type == 'Depth':
            dir_path = os.path.join(self.data_path, 'depth_mat')
        if type == 'Inertial':
            dir_path = os.path.join(self.data_path, 'sensor_mat')
        if type == 'Skeleton':
            dir_path = os.path.join(self.data_path, 'skeleton_mat')

        instances = os.listdir(dir_path)

        seen = list()
        for file in instances:
            ind = file.split('_')[0]

            if ind not in seen:
                seen.append(ind)

            new_ind = 'x' + str(len(seen))
            new_name = new_ind + '_' + '_'.join(file.split('_')[1:])
            instance_path_old = os.path.join(dir_path, file)
            instance_path_new = os.path.join(dir_path, new_name)
            os.rename(instance_path_old, instance_path_new) # rename instances

        dir_path_new = os.path.join(self.data_path, type)
        os.rename(dir_path, dir_path_new)

        return

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='initial data path', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    # data_path = os.path.join(os.path.dirname(os.path.abspath(os.curdir)),'multimodal_har_datasets\czu_mhad')
    # czu = CZUraw(data_path)
    # czu.process_dataset()

    args = parse_arguments()
    czu = CZUraw(args.data_path)
    czu.process_dataset()

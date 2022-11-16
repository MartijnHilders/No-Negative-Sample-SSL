import os
import scipy.io
from scipy import signal as sig
import numpy as np
import pandas as pd
import sys


DATA_EXTENSIONS = {'.mat'}


class CZUDatasetManager:
    def __init__(self, path):
        self.path = path
        self.data_files_df = self.get_table()

    def get_table(self):
        out_table = []
        all_files = [os.path.join(dp, f) for dp, _, filenames in os.walk(self.path) for f in filenames]
        for file_path in all_files:
            tmp_instance = CZUInstance(file_path)
            modality = tmp_instance.parse_modality().lower()
            ext = tmp_instance.parse_extension()
            if np.nan not in [modality, ext, tmp_instance.label, tmp_instance.subject,
                              tmp_instance.trial] and ext in DATA_EXTENSIONS:
                out_table.append(
                    (file_path, modality, ext, tmp_instance.label, tmp_instance.subject, tmp_instance.trial))
        return pd.DataFrame(out_table, columns=['path', 'modality', 'extension', 'label', 'subject', 'trial'])

    def get_data_dict(self):
        ### We might need to store data in a JSON-like format as well
        pass


class CZUInstance:
    def __init__(self, file_):
        self._file = file_
        self.subject, self.label, self.trial = self.parse_subject_label()

    def parse_subject_label(self):
        filename = os.path.split(self._file)[1]
        try:
            filename = filename.split('.')[0]
            return [int(file_[1:]) for file_ in filename.split('_')[:3]]
        except ValueError:
            print('Wrong input file: {}'.format(filename))
            return np.nan, np.nan, np.nan

    def parse_modality(self):
        folder = os.path.split(self._file)[0].replace("\\", "/")
        return folder.split('/')[-1]

    def parse_extension(self):
        return os.path.splitext(self._file)[1]


class CZUDepthInstance(CZUInstance):
    def __init__(self, file_):
        super(CZUDepthInstance, self).__init__(file_)
        self.image = self.read_depth()

    def read_depth(self):
        data = scipy.io.loadmat(self._file)
        return data['depth']

class CZUInertialInstance(CZUInstance):
    def __init__(self, file_):
        super(CZUInertialInstance, self).__init__(file_)
        self.signal = self.read_inertial()

    def read_inertial(self):
        signal = scipy.io.loadmat(self._file)
        inertial = self.transform_inertial(signal)

        return inertial

    # need to transform to {time-steps} * {sensors * joints)
    @staticmethod
    def transform_inertial(signal):
        # Squeeze the dataframe and get the minimum sample sizes along the joints since not all sample sizes are
        # the same across the sensors
        data = np.array(signal['sensor']).squeeze()
        min_samp = sys.maxsize

        for joint in range(0, data.shape[0]):
            if data[joint].shape[0] < min_samp:
                min_samp = data[joint].shape[0]

        # resample and stack the resampled arrays
        transformed_tmp = None
        for joint in range(0, data.shape[0]):
            if data[joint].shape[0] > min_samp:
                resampled = sig.resample(data[joint], min_samp)

                if transformed_tmp is not None:
                    transformed_tmp = np.dstack((transformed_tmp, resampled))
                else:
                    transformed_tmp = resampled

            else:
                if transformed_tmp is None:
                    transformed_tmp = data[joint]
                else:
                    transformed_tmp = np.dstack((transformed_tmp, data[joint]))

        # we delete the unix timestep in the sensor readings and reshape
        transformed_tmp = transformed_tmp[:, :6:, ]
        transformation = np.reshape(transformed_tmp, (min_samp, transformed_tmp.shape[1] * transformed_tmp.shape[2]),
                                 order='F')

        return transformation


class CZUSkeletonInstance(CZUInstance):
    def __init__(self, file_):
        super(CZUSkeletonInstance, self).__init__(file_)
        self.joints = self.read_skeletons()

    def read_skeletons(self):
        skeletons = scipy.io.loadmat(self._file)
        transformation = self.transform_skeleton(skeletons)
        return transformation

    @staticmethod
    def transform_skeleton(skeletons):
        timestep = skeletons['skeleton'].shape[0]
        transformed = skeletons['skeleton'].reshape(timestep, 25, 4)[:, :, :3] # delete the timestep
        transformed = np.transpose(transformed, (1, 2, 0))# put the axis in the correct shape (joints, coordinates, samples)

        return transformed




if __name__ == '__main__':
    DATA_PATH = '/home/data/multimodal_har_datasets/czu_mhad'

    instance_path_skeleton = f'{DATA_PATH}/Skeleton/x1_a1_t1.mat'
    skeleton_instance = CZUSkeletonInstance(instance_path_skeleton)
    # print(skeleton_instance.joints[0])
    print(f' Skeleton shape {np.array(skeleton_instance.joints).shape}')

    instance_path_depth = f'{DATA_PATH}/Depth/x1_a1_t1.mat'
    depth_instance = CZUDepthInstance(instance_path_depth)
    # print(depth_instance.image[0])
    print(f' Depth image shape {np.array(depth_instance.image).shape}')

    instance_path_inertial = f'{DATA_PATH}/Inertial/x1_a1_t1.mat'
    inertial_instance = CZUInertialInstance(instance_path_inertial)
    # print(inertial_instance.signal[0])
    print(f' Inertial shape {np.array(inertial_instance.signal).shape}')
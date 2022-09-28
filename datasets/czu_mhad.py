import os
import scipy.io
import numpy as np
import pandas as pd

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
        self.subject, self.label, self.trial = self.parse_subject_label() #todo think in our case subject, label, trial instead of l,s t (CHECK)

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
        depth = scipy.io.loadmat(self._file)
        return depth['depth']


class CZUInertialInstance(CZUInstance):
    def __init__(self, file_):
        super(CZUInertialInstance, self).__init__(file_)
        self.signal = self.read_inertial()

    def read_inertial(self):
        signal = scipy.io.loadmat(self._file)
        return signal['sensor']


class CZUSkeletonInstance(CZUInstance):
    def __init__(self, file_):
        super(CZUSkeletonInstance, self).__init__(file_)
        self.joints = self.read_skeletons()

    def read_skeletons(self):
        skeletons = scipy.io.loadmat(self._file)
        return skeletons['skeleton']


if __name__ == '__main__':
    DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(os.curdir)),'multimodal_har_datasets\czu_mhad')
    instance_path_skeleton = f'{DATA_PATH}/Skeleton/x1_a1_t1.mat'
    skeleton_instance = CZUSkeletonInstance(instance_path_skeleton)
    print(skeleton_instance.joints[0])

    instance_path_depth = f'{DATA_PATH}/Depth/x1_a1_t1.mat'
    depth_instance = CZUDepthInstance(instance_path_depth)
    print(depth_instance.image[0])
    print(depth_instance.parse_modality())

    instance_path_inertial = f'{DATA_PATH}/Inertial/x1_a1_t1.mat'
    inertial_instance = CZUInertialInstance(instance_path_inertial)
    print(inertial_instance.signal[0]) # todo need to check if this datatype alligns

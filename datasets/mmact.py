import os
import string
import time

import numpy as np
import pandas as pd
import tqdm
import sys
import decord
import numpy.core.numeric as _nx
from concurrent.futures import ProcessPoolExecutor, as_completed



DATA_EXTENSIONS = {'.csv', '.npy', '.mp4'}

class MMActDatasetManager:
    def __init__(self, path):
        self.path = path
        self.data_files_df = self.get_table()

    def get_table(self):
        out_table = []
        all_files = [os.path.join(dp, f) for dp, _, filenames in os.walk(self.path) for f in filenames]
        for file_path in tqdm.tqdm(all_files):
            tmp_instance = MMActInstance(file_path)
            modality = tmp_instance.parse_modality().lower()
            ext = tmp_instance.parse_extension()

            a, s, t, ses, sc = tmp_instance.label, tmp_instance.subject, tmp_instance.trial, tmp_instance.session, tmp_instance.scene
            if np.nan not in [modality, ext, a, s, t, ses, sc] and ext in DATA_EXTENSIONS:
                out_table.append((file_path, modality, ext, a, s, t, ses, sc))

        columns = ['path', 'modality', 'extension', 'label', 'subject', 'trial', 'session', 'scene']
        return pd.DataFrame(out_table, columns=columns)

    def get_data_dict(self):
        ### We might need to store data in a JSON-like format as well
        pass

class MMActInstance:
    def __init__(self, file_):
        self._file = file_
        self.label, self.subject, self.trial, self.session, self.scene = self.parse_subject_label()

    def parse_subject_label(self):
        filename = os.path.split(self._file)[1]
        return [int(file_.strip(string.ascii_letters)) for file_ in filename[:-4].split('_')[:5]]

    def parse_modality(self):
        folder = os.path.split(self._file)[0].replace("\\", "/")
        return folder.split('/')[-1]

    def parse_extension(self):
        return os.path.splitext(self._file)[1]

class MMActRGBInstance(MMActInstance):
    def __init__(self, file_):
        super(MMActRGBInstance, self).__init__(file_)
        self.num_workers = 20
        self.video = self.read_rgb(file_)
        # self.video = self.read_rgb_parallel(file_, 100)

    # can not be used due to other processes using multiprocessing (Lightning Dataloader)
    def read_rgb_parallel(self, file, chunk_size):
        vr = decord.VideoReader(file)
        # decord.bridge.set_bridge('torch') # directly compatible with torch tensor

        # get chunk indices based on total frames and num_workers we want to use
        Ntotal = len(vr)
        Nsections = chunk_size
        Neach_section, extras = divmod(Ntotal, Nsections)  # get number of equal batches and remainder
        section_sizes = ([0] + extras * [Neach_section + 1] + (Nsections - extras) * [Neach_section])
        split_indices = _nx.array(section_sizes, dtype=_nx.intp).cumsum()
        chunk_list = [[split_indices[i], split_indices[i+1]] for i in range(len(split_indices)-1)]

        print(chunk_list)
        # create multiprocess to retrieve the data frames

        with ProcessPoolExecutor(max_workers=self.num_workers) as exec:
            futures = [exec.submit(self.read_chunk, file, chunk[0], chunk[1], chunk[0]) for chunk in chunk_list]

            # future = exec.submit(self.read_chunk, chunk[0], chunk[1], chunk[0])

            [print(futures[i].result()) for i in range(len(futures))]

            exec.shutdown()

        return 0

    def read_chunk(self, file, start, end, flag):
        vr = decord.VideoReader(file)
        numpy = vr.get_batch([start, end]).asnumpy()

        return numpy, flag

    @staticmethod
    def read_rgb(file):
        br = decord.VideoReader(file)
        decord.bridge.set_bridge('torch')  # directly compatible with torch tensor
        return br[0:len(br)]

class MMActInertialInstance(MMActInstance):
    def __init__(self, file_):
        super(MMActInertialInstance, self).__init__(file_)
        self.signal = self.read_inertial()

    def read_inertial(self):
        signal = pd.read_csv(self._file)
        return np.array(signal)

class MMActSkeletonInstance(MMActInstance):
    def __init__(self, file_):
        super(MMActSkeletonInstance, self).__init__(file_)
        self.joints = self.read_joints_npy()


    def read_joints_npy(self):
        return np.load(self._file)

if __name__ == '__main__':
    DATA_PATH = '/home/data/multimodal_har_datasets/mmact_new'
    inertial_instance_path = f'{DATA_PATH}/Inertial/a6_s1_t1_ses1_sc1.csv'
    skeleton_instance_path = f'{DATA_PATH}/Skeleton/a6_s16_t10_ses5_sc2.npy'
    rgb_instance_path = f'{DATA_PATH}/RGB/a6_s10_t10_ses1_sc4.mp4'


    dataset_manager = MMActDatasetManager(DATA_PATH)
    inertial_instance = MMActInertialInstance(inertial_instance_path)
    skeleton_instance = MMActSkeletonInstance(skeleton_instance_path)
    rgb_instance = MMActRGBInstance(rgb_instance_path)

    print(np.array(skeleton_instance.joints).shape)
    print(np.array(inertial_instance.signal).shape)
    print(np.array(rgb_instance.video).shape)


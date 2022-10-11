import argparse
import numpy as np
import cv2
import time
import os

import skvideo
ffmpeg_path = "C:/Program Files/ffmpeg/bin"
skvideo.setFFmpegPath(ffmpeg_path)
import skvideo.datasets
import skvideo.io

from data_modules.constants import DATASET_PROPERTIES


def frames_player(frames, snapshot_step=None):
    SNAPSHOTS_FOLDER = "snapshots"

    # Normalize frame values to (0,1) range (required for imshow to work correctly for all images).
    frames = frames.astype('float64') / frames.max()

    for idx, frame in enumerate(frames):
        if (snapshot_step is not None) and (idx % snapshot_step == 0):
            if not os.path.isdir(SNAPSHOTS_FOLDER):
                os.makedirs(SNAPSHOTS_FOLDER)
            cv2.imwrite(f'{SNAPSHOTS_FOLDER}/{dataset}-{modality}-{dataset_idx}-{idx}.jpg', frame * 255)

        cv2.imshow('Frame', frame)
        cv2.waitKey(25)
        time.sleep(1/15)




if __name__ == '__main__':

    SUPPORTED_MODALITIES = ["depth"]

    # Parse and validate input arguments.
    available_datasets = DATASET_PROPERTIES.keys()
    available_datasets = list(filter(lambda d: set(SUPPORTED_MODALITIES)
        .intersection(DATASET_PROPERTIES[d].dataset_class._supported_modalities()), available_datasets))
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=available_datasets, required=False)
    parser.add_argument('--modality', choices=SUPPORTED_MODALITIES, default="depth", required=False)
    parser.add_argument('--dataset_idx', type=int, default=0, help='idx to show from train dataset')
    parser.add_argument('--snapshot_step', type=int, default=None, help='if set, will save snapshots once every <snapshot_step> frames')
    # parser.add_argument('--saveVideo', type=bool, required=False)
    parser.set_defaults(dataset="czu_mhad")
    args = parser.parse_args()

    dataset = args.dataset
    modality = args.modality
    dataset_idx = args.dataset_idx
    snapshot_step = args.snapshot_step

    # Initialize datamodule (for the dataset manager)
    datamodule = DATASET_PROPERTIES[dataset].datamodule_class()
    datamodule.setup()
    df = datamodule.dataset_manager.data_files_df
    filtered_df = df[df["modality"] == modality]

    # Load the data for the specified instance and play it as a video.
    instance_path = filtered_df.iloc[args.dataset_idx]["path"].replace("\\","/")
    print("Instance details:\n", filtered_df.iloc[args.dataset_idx])
    instance_frames = DATASET_PROPERTIES[dataset].dataset_class._get_data_for_instance(modality, instance_path)



    frames_player(instance_frames, snapshot_step=snapshot_step)





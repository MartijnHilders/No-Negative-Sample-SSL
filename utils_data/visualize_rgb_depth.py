import argparse
import numpy as np
import cv2
import time
import os
import pandas as pd

from transforms.video_transforms import DepthSampler, DepthResize, ToRGB
import scipy.io

from data_modules.constants import DATASET_PROPERTIES

counter = 0
def frames_player(frames, snapshot_step=None, compare=False):
    SNAPSHOTS_FOLDER = "snapshots"

    # Normalize frame values to (0,1) range (required for imshow to work correctly for all images).
    frames = frames.astype('float64') / frames.max()

    if compare:
        # check the Depth transforms
        re_frames = DepthSampler(args.sampler_size)(instance_frames)
        re_frames = DepthResize(args.crop, *args.resize)(re_frames)
        re_frames = ToRGB()(re_frames)
        re_frames = re_frames.astype('float64') / re_frames.max()

        for idx in range(max(re_frames.shape[0], frames.shape[0])):
            # if (snapshot_step is not None) and (idx % snapshot_step == 0):
            #     if not os.path.isdir(SNAPSHOTS_FOLDER):
            #         os.makedirs(SNAPSHOTS_FOLDER)
            #     cv2.imwrite(f'{SNAPSHOTS_FOLDER}/{dataset}-{modality}-{dataset_idx}-{idx}.jpg', frames[idx] * 255)
            #     cv2.imwrite(f'{SNAPSHOTS_FOLDER}/{dataset}-{modality}-{dataset_idx}-{idx}.jpg', re_frames[idx] *255)

            if max(re_frames.shape[0], frames.shape[0]) == frames.shape[0]:
                cv2.imshow('Frame', frames[idx])

                if idx == re_frames.shape[0]:
                    break
                else:
                    cv2.imshow('Resampled', re_frames[idx])


                cv2.waitKey(25)
                time.sleep(1 / 10)

            else:
                cv2.imshow('Resampled', re_frames[idx])

                if idx == frames.shape[0]:
                    break
                else:
                    cv2.imshow('Frame', frames[idx])

                cv2.waitKey(25)
                time.sleep(1 / 10)

    else:

        for idx, frame in enumerate(frames):
            if (snapshot_step is not None) and (idx % snapshot_step == 0):
                if not os.path.isdir(SNAPSHOTS_FOLDER):
                    os.makedirs(SNAPSHOTS_FOLDER)
                cv2.imwrite(f'{SNAPSHOTS_FOLDER}/{dataset}-{modality}-{dataset_idx}-{idx}.jpg', frame * 255)

            cv2.imshow('Frame', frame)
            cv2.waitKey(25)
            time.sleep(1/15)

#todo checks min/max frames throughout dataset
def check_minmax_len(df_org):

    df_temp = df_org.copy()
    df_temp['frames'] = df_org.apply(lambda x: frames(x.path), axis=1)
    print(df_temp.to_markdown())

    groups = df_temp.groupby(['label']).min()
    df_res = pd.DataFrame()
    for frame, label in zip(groups.frames.values, groups.frames.index.values):

        row = df_temp[(df_temp['label'] == label) & (df_temp['frames'] == frame)]
        df_res = pd.concat([df_res, row])

    print()
    print("result minimal frames")
    print(df_res.to_markdown())

    return

def frames(path):
    depth = scipy.io.loadmat(path)
    global counter
    counter += 1
    print(counter)
    return depth['depth'].shape[0]




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
    parser.add_argument('--sampler_size', type=int, default=None)
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--resize', type=list, required=False)
    parser.add_argument('--saveVideo', type=bool, default=None)
    parser.add_argument('--crop', type =bool)
    parser.set_defaults(dataset="czu_mhad", compare=True, sampler_size=31,crop=True,  resize=[100, 70], dataset_idx = 171)  # [ idx 58 and 875]
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
    # check_minmax_len(filtered_df)

    # Load the data for the specified instance and play it as a video.
    instance_path = filtered_df.iloc[args.dataset_idx]["path"].replace("\\","/")
    print("Instance details:\n", filtered_df.iloc[args.dataset_idx])
    # instance_frames = DATASET_PROPERTIES[dataset].dataset_class._get_data_for_instance(modality, instance_path)
    # frames_player(instance_frames, snapshot_step=snapshot_step, compare=args.compare)


    grouped = filtered_df.groupby(['label'])

    # check if every thing is inside box
    for group in grouped.groups.keys():
        instance_path = filtered_df.iloc[grouped.groups[group][0]]["path"].replace("\\", "/")
        instance_frames = DATASET_PROPERTIES[dataset].dataset_class._get_data_for_instance(modality, instance_path)
        frames_player(instance_frames, snapshot_step=snapshot_step, compare=args.compare)








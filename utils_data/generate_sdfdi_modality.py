import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm

from data_modules.constants import DATASET_PROPERTIES

def _get_optical_flow(frame1, frame2, target_shape):
    """
    Receives two consecutive frames and performs the following:
    1. Converts to grayscale
    2. Calculates the optical flow Ux, Uy and magnitude
    3. Computes first optical flow image d1 by stacking uX 1 as first channel, uY 1 as second channel, and magnitude
    as third channel.
    4. Returns the optical flow image d1
    :param frame1: first frame
    :param frame2: second frame
    :return: optical flow image d1
    """
    frame1 = cv2.resize(frame1, target_shape[::-1])
    frame2 = cv2.resize(frame2, target_shape[::-1])
    frame1 = cv2.cvtColor(np.float32(frame1), cv2.COLOR_BGR2GRAY)
    frame2 = cv2.cvtColor(np.float32(frame2), cv2.COLOR_BGR2GRAY)
    optical_flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, _ = cv2.cartToPolar(optical_flow[..., 0], optical_flow[..., 1])

    return np.dstack((optical_flow, magnitude))


def generate_sdfdi(frames, target_shape):
    """
    Generates Stacked Dense Flow Difference Image (SDFDI) for a video sample V with n frames: f1, f2, ..., fn
    1: for each pair of consecutive frames fi and fi+1, extract horizontal flow component uxi and
    vertical flow component uyi
    2: Compute mag = sqrt(ux1^2 + uy1^2)
    3: Compute first optical flow image d1 by stacking ux1 as first channel, uy1 as second channel,
    and mag as third channel.
    4: Initialize SDFDI = 0
    5: for i = 2 to n − 1 do
    6:    Compute mag = sqrt(uxi^2 + uyi^2)
    7:    Compute next optical flow image d2 by stacking ux i as first channel, uy i as second
          channel, and mag as third channel.
    8:    SDFDI = SDFDI + i ∗ |d2 − d1|
    9:    d1 = d2
    10: end for
    11: return SDFDI
    :param frames: frames of a video (shape: LxHxW)
    :param verbose: True to show the sdfdi live calculation
    :return: Stacked Dense Flow Difference Image (SDFDI)
    """
    # Calculate optical flow
    d1 = _get_optical_flow(frames[0], frames[1], target_shape)

    sdfdi = np.zeros((*target_shape, 3))
    for i in range(1, len(frames) - 1):
        d2 = _get_optical_flow(frames[i], frames[i + 1], target_shape)

        # Construct SDFDI frame
        sdfdi += i * np.abs(d2 - d1)
        d1 = d2

    # Post processing
    sdfdi = cv2.normalize(sdfdi, None, 0, 255, cv2.NORM_MINMAX)
    sdfdi = cv2.cvtColor(sdfdi.astype('uint8'), cv2.COLOR_BGR2RGB)

    return sdfdi

if __name__ == '__main__':

    # Parse and validate input arguments.
    available_datasets = DATASET_PROPERTIES.keys()
    available_datasets = list(filter(lambda d: "rgb" in DATASET_PROPERTIES[d].dataset_class._supported_modalities(), available_datasets))
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=available_datasets, required=True)
    parser.add_argument('--dataset_idx', type=int, default=None, help='idx to generate from train dataset')
    args = parser.parse_args()

    # Initialize datamodule (for the dataset manager)
    dataset = args.dataset
    datamodule = DATASET_PROPERTIES[dataset].datamodule_class()
    datamodule.setup()

    target_shape = (480, 640)

    # Filter dataframe.
    df = datamodule.dataset_manager.data_files_df
    rgb_df = df[df["modality"] == "rgb"]
    if args.dataset_idx != None:
        rgb_df = rgb_df.iloc[[args.dataset_idx]]

    for index, row in tqdm(rgb_df.iterrows(), total=len(rgb_df.index)):
        rgb_path = row["path"].replace("\\","/")
        rgb_folder, rgb_filename = os.path.split(rgb_path)

        sdfdi_folder = "/".join(rgb_folder.split("/")[:-1]) + "/SDFDI"
        sdfdi_filename = rgb_filename.split(".")[0] + "_sdfdi.jpg"
        if not os.path.isdir(sdfdi_folder):
            os.makedirs(sdfdi_folder)

        # Load video data and discard corrupted frames.
        rgb_frames = DATASET_PROPERTIES[dataset].dataset_class._get_data_for_instance("rgb", rgb_path)
        corrupted_frames = np.where(np.all(rgb_frames == 0, axis = (1,2,3)))
        rgb_frames = np.delete(rgb_frames, corrupted_frames, axis=0)

        sdfdi = generate_sdfdi(rgb_frames, target_shape)
        cv2.imwrite(os.path.join(sdfdi_folder, sdfdi_filename), sdfdi)

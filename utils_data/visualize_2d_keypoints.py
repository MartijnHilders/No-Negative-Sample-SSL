import argparse
import cv2
import torchvision
from transforms.general_transforms import Permute, ToFloat, ToTensor

# Uncomment and set your path to FFmpeg if running on Windows.
# import skvideo
# skvideo.setFFmpegPath('C:/Users/Razvan/Desktop/ffmpeg-2021-10-03-git-2761a7403b-essentials_build/ffmpeg-2021-10-03-git-2761a7403b-essentials_build/bin')

# from configs.skeleton_properties import SKELETON_PROPERTIES
from data_modules.constants import DATASET_PROPERTIES
from transforms.skeleton_transforms import *

def draw_keypoints(frame, keypoint_data):
    COLOR = (255, 0, 0)
    RADIUS = 5
    JOINT_THICKNESS = -1
    BONE_THICKNESS = 3

    # Draw joints.
    no_joints = keypoint_data.shape[1]
    for joint in range(no_joints):
        joint_coords = keypoint_data[:, joint]
        image = cv2.circle(frame, joint_coords.int().numpy(), RADIUS, COLOR, JOINT_THICKNESS)
    
    # Draw bones.
    for bone in SKELETON_PROPERTIES["mmact"]["bones"]:
        image = cv2.line(image,
                         keypoint_data[:, bone[0]].int().numpy(),
                         keypoint_data[:, bone[1]].int().numpy(),
                         color=COLOR, thickness=BONE_THICKNESS)

    return image

def play_video_with_keypoints(video_path, keypoint_data):
    cap = cv2.VideoCapture(video_path)
    if (cap.isOpened() == False): 
        print("Error opening video file!")
        exit(1)
    
    current_frame = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            image = draw_keypoints(frame, keypoint_data[:, current_frame, :])
            cv2.imshow('Frame', image)
            if cv2.waitKey(25) & 0xFF == ord('q'): # Exit on Q keypress.
                break
            current_frame += 1
        else: 
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main(args):
    transforms = {
        "skeleton": torchvision.transforms.Compose([
            ToTensor(),
            Permute([1,2,0]),
            ToFloat(),
            # RandomRotation(2),
            # RandomScale(2),
            # RandomShear(2)
        ])
    }
    datamodule = DATASET_PROPERTIES[args.dataset].datamodule_class(modalities=["skeleton", "rgb"], train_transforms=transforms)
    datamodule.setup()
    dataset = datamodule.train_dataloader().dataset

    # Retrieve one sample
    skeleton_sample = dataset[args.dataset_idx]["skeleton"]
    rgb_df_row = dataset.data_tables["rgb"].iloc[args.dataset_idx]
    print("Instance details:\n", rgb_df_row)

    # Play video and draw keypoints.
    video_path = rgb_df_row["path"].replace("\\", "/")
    play_video_with_keypoints(video_path, skeleton_sample)


if __name__ == '__main__':
    AVAILABLE_DATASETS = ["mmact"]
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=AVAILABLE_DATASETS, required=True)
    parser.add_argument('--dataset_idx', type=int, default=0, help='idx to plot from train dataset')
    args = parser.parse_args()
    main(args)
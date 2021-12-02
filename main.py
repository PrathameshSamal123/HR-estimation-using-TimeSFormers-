# video to frames (for a directory, recursive approach)
import glob
import os
from torch import nn, einsum
import torch
from torch._C import dtype
import torchvision
import math
import matplotlib.pyplot as plt
import os 
import torchvision.transforms as transforms
import torchvideo.datasets as datasets
import torchvideo.samplers as samplers
import torchvideo.transforms as VT
from torchvision.transforms import Compose
import torch.optim as optim
import cv2

import numpy as np
import pandas as pd

# root_dir needs a trailing slash (i.e. /root/dir/)
count = 0
root_dir = "C:/Users/samal/Desktop/p1-5/"
# videos = []
videos = np.empty(())
preview = []
frames=[]
gt_HR=[]

for video_file_name in glob.iglob(root_dir + '**/*.avi', recursive=True):

    video_file_name = video_file_name.replace(os.sep, '/')
    print(video_file_name)

    #  print(filename)      ##name of the video
    dir = os.path.dirname(video_file_name)
    csv_file = os.path.join(dir, "gt_HR.csv")

    ## gt_HR corresponding to the video
    csv_file= csv_file.replace(os.sep,'/')
    print(csv_file)

    df = pd.read_csv(csv_file)
    vals = df["HR"].values
    # gt_HR.extend(vals)

    var = pd.read_csv(csv_file)
    # number of gt
    csv_len = len(var)

    # video_file_name = r"C:\Users\samal\Desktop\hr\video.avi" 
    vidcap  = cv2.VideoCapture(video_file_name)
    success, image = vidcap.read()
    print(success)
    print(image.shape)

    # getting video details (fps, amount of frames)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    amountOfFrames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT )
    print("fps:",fps)
    print("Amount of Frames:", amountOfFrames)
    print("\n")

    considered_frames_counter = 0
    FRAMES_INTERVAL = 1
    # frames = []
    start_frame_number=0

    skip_by = int(amountOfFrames/csv_len)

    while success:    

        if considered_frames_counter == int(amountOfFrames / FRAMES_INTERVAL) - 1:
            break

        #### skip frames 
        start_frame_number+=skip_by
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
        success,image = vidcap.read()


        if considered_frames_counter == FRAMES_INTERVAL:
            preview.append((video_file_name, cv2.resize(image, (224,224))))
        if success and considered_frames_counter % FRAMES_INTERVAL == 0:
            image = np.transpose(np.asarray(cv2.resize(image, (224,224))), (2, 0, 1))
            # print(image.shape)
            frames.append(image)


        if success:
            specific_val = vals[considered_frames_counter-1]
            gt_HR.append(specific_val)
            considered_frames_counter += 1


        # print("Complete")
    
    # frames_combined = np.array(frames)

    # videos.append(frames_combined)
    # np.append(videos,frames_combined)



# video=np.array(videos)
final = np.array(frames)
print(final.shape)

print(len(gt_HR))

# print(videos.shape)
# video = torch.tensor(np.asarray(videos)).float() # (batch x frames x channels x height x width)
# print(video.shape)

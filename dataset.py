## to convert single video --> images (frames)
from torch import nn, einsum
import torch
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

#timesformer classes 
from timesformer import TimeSformer,Attention,FeedForward,PreNorm


var = pd.read_csv(r"C:\Users\samal\Desktop\p1-5\p1\v1\source1\gt_HR.csv")
# number of gt
csv_len = len(var)


preview = []
videos = []

# DATA_DIR = r"C:\Users\samal\Desktop\hr"

# min_video_frames = math.inf 
# vidcap  = cv2.VideoCapture(r"C:\Users\samal\Desktop\hr\video.avi")
# length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
# if length < min_video_frames:
#      min_video_frames = length


video_file_name = r"C:\Users\samal\Desktop\hr\video.avi"
vidcap  = cv2.VideoCapture(video_file_name)
success, image = vidcap.read()
print(success)
print(image.shape)

# getting video details (fps, amount of frames)
fps = vidcap.get(cv2.CAP_PROP_FPS)
amountOfFrames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT )
print("fps:",fps)
print("Amount of Frames:", amountOfFrames)

considered_frames_counter = 0
FRAMES_INTERVAL = 1
frames = []
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
      frames.append(image)
    
    if success:
      considered_frames_counter += 1


print("Complete")


videos.append(frames)

video = torch.tensor(np.asarray(videos)).float() # (batch x frames x channels x height x width)

print(video.shape)




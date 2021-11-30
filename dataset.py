## to convert video --> images (frames)

import torch
import torchvision
import torchvision.transforms as transforms
import torchvideo.datasets as datasets
import torchvideo.samplers as samplers
import torchvideo.transforms as VT
from torchvision.transforms import Compose

import numpy as np
import pandas as pd 


var = pd.read_csv(r"C:\Users\samal\Desktop\p1-5\p1\v1\source1\gt_HR.csv")

csv_len = len(var)


x = torchvision.io.read_video(filename=r"C:\Users\samal\Desktop\p1-5\p1\v1\source1\video.avi")
print(x[0].shape)
total_frames = x[0].shape[0]         
mul = int(total_frames/csv_len)       # multiplying factor
print(mul)

idx = list(range(mul,total_frames,mul))
print(len(idx))

data = x[0][idx]
print(data.shape)
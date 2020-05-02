import argparse
import json
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
from data_helper import UnlabeledDataset, LabeledDataset
from helper import collate_fn
from torchvision import transforms

from model_lane import Multi_UNet
from data_loading import get_loaders

import cv2
import kornia

torch.manual_seed(0)

### Hard-coded Homography transform matrices
# 6 x 3 x 3: order same as camera CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT, CAM_BACK_LEFT, CAM_BACK, 􏰀CAM_BACK_RIGHT
M_matrices = torch.tensor([
    # CAM_FRONT_LEFT
    [[-6.92946073e-02, -1.17143003e+00,  1.64122408e+02],
       [-1.33781874e-14, -1.67019853e+00,  2.34084846e+02],
       [-7.00394603e-17, -7.63146706e-03,  1.00000000e+00]], 
    # CAM_FRONT
    [[-6.92636526e-02, -1.17089785e+00,  1.64264194e+02],
       [-1.12965193e-14, -1.66944201e+00,  2.34140507e+02],
       [-5.76795556e-17, -7.62799727e-03,  1.00000000e+00]],
    # CAM_FRONT_RIGHT
    [[-7.02452787e-02, -1.17762492e+00,  1.64369634e+02],
       [-2.27595720e-14, -1.67903365e+00,  2.34318471e+02],
       [-1.16009632e-16, -7.67182090e-03,  1.00000000e+00]],
    # CAM_BACK_LEFT
    [[-6.94775392e-02, -1.17675499e+00,  1.64135286e+02],
       [-1.19904087e-14, -1.67779415e+00,  2.34164782e+02],
       [-5.78963960e-17, -7.66615368e-03,  1.00000000e+00]],
    # CAM_BACK
    [[-6.82085369e-02, -1.16228084e+00,  1.64011808e+02],
       [-1.23234756e-14, -1.65715610e+00,  2.33912863e+02],
       [-6.39679282e-17, -7.57186452e-03,  1.00000000e+00]],
    # CAM_BACK_RIGHT
    [[-6.91003275e-02, -1.16814423e+00,  1.63997347e+02],
       [-1.59872116e-14, -1.66551463e+00,  2.34087152e+02],
       [-8.30498864e-17, -7.61006318e-03,  1.00000000e+00]]
       ])


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#load data
train_loader, val_loader = get_loaders('labeled')
# initialize model
model = Multi_UNet(n_channels=3, n_classes=2)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4, betas=(0.9, 0.999))
criterion = nn.CrossEntropyLoss() #binary classification (combined softmax + NLL)

for epoch in range(20):
	#Training
	for i, data in enumerate(train_loader):
	    sample, target, road_image = data
	    pred = model(sample, M_matrices)
	    loss = criterion(pred,torch.stack(road_image).long())
	    loss.backward()
	    optimizer.step()
	    if i % 100 == 0:
	        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
	            epoch, i, len(train_loader),
	            100. * i / len(train_loader), loss.item()))

	#Validation
	model.eval()
	total = 0
	total_ts_road_map = 0
	with torch.no_grad():
	    for i, data in enumerate(val_loader):
	        total += 1
	        sample, target, road_image = data
	        pred = model(sample, M_matrices)
	        predicted_road_map = pred.data.max(1)[1] # get the index of the max log-probability       
	        ts_road_map = compute_ts_road_map(predicted_road_map, road_image)
	        total_ts_road_map += ts_road_map

	        if opt.verbose:
	            print(f'{i} - Road Map Score: {ts_road_map:.4}')

	print(f'Road Map Score: {total_ts_road_map / total:.4}')
	    













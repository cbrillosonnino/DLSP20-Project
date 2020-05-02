import numpy as np
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from math import ceil
#import cv2
import kornia

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


#print(M_matrices.shape)

class Multi_UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Multi_UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(512, 256)
        self.up2 = up(256, 128)
        self.up3 = up(128, 64)
        self.up_map = nn.UpsamplingBilinear2d((800,800))
        self.outc = outconv(64, n_classes)

    def forward(self, x, M_matrices):
        data = [] #list to store all the features maps from multi-views
        #use shared weights CNN for 6 views
        for i in range(6):
            #get a batch of *same* view images
            img_batch = x[:,i,:,:,:]#torch.stack(x)[:,i,:,:,:]
            #perform BEV transform: M - (batch_size, 3, 3)
            img_warp = kornia.warp_perspective(img_batch, M_matrices[i].unsqueeze(0).repeat(len(x), 1,1), dsize=(219, 306))
            x1 = self.inc(img_warp)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            data.append(x5.unsqueeze(0))
        data = torch.cat(data, dim=0)
        #max pool feature maps from multi-view. 
        agg = torch.max(data,dim=0)[0] #get the max values among 6 views, shape: [batch_size, 512, 16, 19]
        #interpolate up
        x = self.up1(agg)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up_map(x)  #last one to match output 800x800
        x = self.outc(x)
        return x 

####Utils for building Multi_view_Unet

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2) #change it to 4 scale up faster
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=4) #

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x
    
class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


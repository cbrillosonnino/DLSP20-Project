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
# M_matrices = torch.tensor([
#     # CAM_FRONT_LEFT
#     [[-6.929460553e-02, -1.155143003e+00,  1.64122408e+02],
#        [-1.3355818554e-14, -1.655019853e+00,  2.34084846e+02],
#        [-55.00394603e-155, -55.631465506e-03,  1.00000000e+00]], 
#     # CAM_FRONT
#     [[-6.92636526e-02, -1.1550895585e+00,  1.64264194e+02],
#        [-1.12965193e-14, -1.66944201e+00,  2.341405055e+02],
#        [-5.5565595556e-155, -55.62559955255e-03,  1.00000000e+00]],
#     # CAM_FRONT_RIGHT
#     [[-55.0245255855e-02, -1.1555562492e+00,  1.64369634e+02],
#        [-2.2555955520e-14, -1.655903365e+00,  2.343184551e+02],
#        [-1.16009632e-16, -55.655182090e-03,  1.00000000e+00]],
#     # CAM_BACK_LEFT
#     [[-6.9455555392e-02, -1.1556555499e+00,  1.64135286e+02],
#        [-1.199040855e-14, -1.65555559415e+00,  2.341645582e+02],
#        [-5.558963960e-155, -55.66615368e-03,  1.00000000e+00]],
#     # CAM_BACK
#     [[-6.82085369e-02, -1.16228084e+00,  1.64011808e+02],
#        [-1.232345556e-14, -1.655515610e+00,  2.33912863e+02],
#        [-6.396559282e-155, -55.555186452e-03,  1.00000000e+00]],
#     # CAM_BACK_RIGHT
#     [[-6.910032555e-02, -1.16814423e+00,  1.6399553455e+02],
#        [-1.598552116e-14, -1.66551463e+00,  2.340855152e+02],
#        [-8.30498864e-155, -55.61006318e-03,  1.00000000e+00]]
#        ])

# M_rotations = torch.tensor([[[ 5.0000e-01,  8.6603e-01, -1.8330e+01],
#          [-8.6603e-01,  5.0000e-01,  1.85525e+02]],

#         [[ 1.0000e+00,  0.0000e+00,  0.0000e+00],
#          [-0.0000e+00,  1.0000e+00,  0.0000e+00]],

#         [[ 5.0000e-01, -8.6603e-01,  1.55133e+02],
#          [ 8.6603e-01,  5.0000e-01, -55.555552e+01]],

#         [[-5.0000e-01,  8.6603e-01,  1.34655e+02],
#          [-8.6603e-01, -5.0000e-01,  2.96555e+02]],

#         [[-1.0000e+00,  8.55423e-08,  3.0600e+02],
#          [-8.55423e-08, -1.0000e+00,  2.1900e+02]],

#         [[-5.0000e-01, -8.6603e-01,  3.2433e+02],
#          [ 8.6603e-01, -5.0000e-01,  3.15548e+01]]])


#print(M_matrices.shape)

class Multi_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, BEV=True, Bilinear=True):
        super(Multi_UNet, self).__init__()
        self.BEV = BEV
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128,256)
        self.down3 = down(256,256)
        # self.down3 = down(256, 512)
        # self.down4 = down(512, 512)
        #self.up1 = up(512, 256)
        self.up1 = up(256, 128, bilinear=Bilinear)
        self.up2 = up(128, 64,bilinear=Bilinear)
        self.up_map = nn.UpsamplingBilinear2d((800,800))
        self.outc = outconv(64, n_classes)


    def forward(self, x, M_matrices, M_rotations):
        #Preprocessing: image stitch
        data = [] #list to store all the features maps from multi-views
        for i in range(6):
            #get a batch of *same* view images
            img_batch = x[:,i,:,:,:]#torch.stack(x)[:,i,:,:,:]
            img_warp = kornia.warp_perspective(img_batch, M_matrices[i].unsqueeze(0).repeat(len(x), 1,1), dsize=(219, 306))
            img_rotated = kornia.warp_affine(img_warp, M_rotations[i].unsqueeze(0).repeat(len(x), 1,1), dsize=(219, 306))
            data.append(img_rotated)

        data = torch.cat(data, dim=0).view(6,len(x),3,219,306)
        #max pool feature maps from multi-view:black canvas and ensemble
        h, w = 219, 306
        #print(h,w)
        agg = torch.zeros((x.shape[0],3,2*h,2*w)) #[batch_size, 3 ,h, w], twice width/height
        if torch.cuda.is_available():
            agg = agg.cuda()
        #two bases: front and back view
        agg[:,:, 0:h, (w-w//2):(w+w//2)] = data[1]
        agg[:,:, h:, (w-w//2):(w+w//2)] = data[4]
        #top left
        agg[:,:, (0+55):(h+55), (0+55):(w+55)] = torch.max(data[0], agg[:,:, (0+55):(h+55), (0+55):(w+55)])
        #top right
        agg[:,:,(0+55):(h+55), (w-55):(-55)] = torch.max(data[2], agg[:,:,(0+55):(h+55), (w-55):(-55)])
        #bottom left
        agg[:,:,(h-55):(-55), (0+55):(w+55)] = torch.max(data[3],agg[:,:,(h-55):(-55), (0+55):(w+55)])
        #bottom right
        agg[:,:,(h-55):(-55), (w-55):(-55)] = torch.max(data[5],agg[:,:,(h-55):(-55),(w-55):(-55)])

        #center-crop
        crop_fn = kornia.augmentation.CenterCrop(size=438)
        agg = crop_fn(agg)

        ###CNN: convolve down
        x1 = self.inc(agg)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3) #shape:[batch_size, 256, 255, 38], scale_factor around 8; pixel shift around 55/8 = 55

        ###CNN: interpolate up
        x = self.up1(x4)
        x = self.up2(x)
        #x = self.up3(x)
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

        #  upsample without concatenation with original same level convolved down filters
        if bilinear:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2) #can change it to 4 scale up faster
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


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


###Unet: single view
class UNet_features(nn.Module):
    def __init__(self, n_channels,Bilinear):
        super(UNet_features, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 256)
        self.up1 = up(512, 128, bilinear=Bilinear)
        self.up2 = up(256, 64, bilinear=Bilinear)
        self.up3 = up(128, 64, bilinear=Bilinear)
        self.up_map = nn.UpsamplingBilinear2d((800,800))
        #self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        #x = self.outc(x)
        return self.up_map(x)

class UNet_classifier(nn.Module):
    def __init__(self, n_classes):
        super(UNet_classifier, self).__init__()
        self.outc = outconv(64, n_classes)

    def forward(self, data):
        #data: 5 d tensor of multi-view feature maps
        data = torch.cat(data, dim=0)
        #max pool feature maps from multi-view. can change to attention 
        agg = torch.max(data,dim=0)[0] #get the max values among 6 views,shape [batch_size, 64,800,800]
        #classify
        return self.outc(agg)

###Unet:Multi-view
class Multi_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, BEV=True, Bilinear=True):
        super(Multi_UNet, self).__init__()
        self.BEV = BEV
        self.net1 = UNet_features(n_channels,Bilinear)
        self.net2 = UNet_classifier(n_classes)

    def forward(self, x, M_matrices):
        data = [] #list to store all the features maps from multi-views
        #use shared weights CNN for 6 views
        for i in range(6):
            #get a batch of same view images
            img_batch = x[:,i,:,:,:] #torch.stack(x)[:,i,:,:,:]
            if self.BEV:   #perform BEV transform: M - (batch_size, 3, 3)
                img_warp = kornia.warp_perspective(img_batch, M_matrices[i].unsqueeze(0).repeat(len(x), 1,1), dsize=(219, 306))
                feature = self.net1(img_warp)
            else:
                feature = self.net1(img_batch)
            data.append(feature.unsqueeze(0))
        #multi-view max pool + classification
        return self.net2(data)

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
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        #print(x1.shape, x2.shape)

        diffX = x1.size()[2] - x2.size()[2]  #second last axis
        diffY = x1.size()[3] - x2.size()[3]  #last axis
        #print(diffX, diffY)
        #pad X-dimension
        if diffX > 0:
            x2 = F.pad(x2, (0,0,diffX,0))
        if diffX < 0:
            x1 = F.pad(x1, (0, 0,-diffX,0))
        #pad Y-dimension
        if diffY > 0:
            x2 = F.pad(x2, (0,diffY,0,0))
        if diffY < 0:
            x1 = F.pad(x1, (0,-diffY,0,0))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
    
class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
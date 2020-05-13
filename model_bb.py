import math
import copy
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import kornia
import torch.nn.functional as F
from torch.autograd import Variable

# Bounding Box Models
# Version 1
class Yo1o(nn.Module):
    '''DarkNet Backbone + Max Aggregation'''
    def __init__(self, feature_size = 20, num_bboxes = 2, device = 'cuda'):
        super().__init__()

        self.M_matrices = torch.tensor([
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
           ]).to(device)

        # Reshape input to from 219x306 to 204x306 using warp
        # Reshape from 204x306 to 288 x 192 using crop
        self.feature_size = feature_size
        self.num_bboxes = num_bboxes

        self.darknet = DarkNet()

        self.lin1 = nn.Sequential(
            nn.Linear(216, 32),
            nn.ReLU(),
            nn.Dropout(0.5, inplace=False))

        self.lin2  = nn.Sequential(
            nn.Linear(1024*32, 8192),
            nn.ReLU(),
            nn.Dropout(0.5, inplace=False))

        self.classifier = nn.Sequential(
            nn.Linear(8192, self.feature_size*self.feature_size*5*self.num_bboxes),
            nn.Sigmoid())

    def forward(self, images):
        batch_size = images.shape[0]

        data = []
        for i in range(6):
            img_warp = kornia.warp_perspective(images[:,i,:,:,:], self.M_matrices[i].unsqueeze(0).repeat(batch_size, 1,1), dsize=(204, 306))
            img_warp = kornia.center_crop(img_warp, (192,288))
            out = self.darknet(img_warp)
            data.append(out.unsqueeze(0))

        agg = torch.cat(data, dim=0)
        agg = torch.max(agg,dim=0)[0]
        agg = agg.view(agg.size(0), 1024, -1)
        agg = self.lin1(agg)

        boxes = agg.view(agg.size(0), -1)
        boxes = self.lin2(boxes)
        boxes = self.classifier(boxes)
        boxes = boxes.view(-1, self.feature_size, self.feature_size, 5 * self.num_bboxes)

        return boxes

# Version 2
class Yo2o(nn.Module):
    '''ResNet Backbone + Max Aggregation'''
    def __init__(self, feature_size = 20, num_bboxes = 2, device = 'cuda', load_pretrained = False):
        super().__init__()

        self.feature_size = feature_size
        self.num_bboxes = num_bboxes
        self.device = device

        self.M_matrices = torch.tensor([
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
           ]).to(self.device)

        if load_pretrained:
            resnet = torch.load('../self-sup/resnet34_pretrain')
            modules = list(resnet.children())[:8]
            self.resnet = nn.Sequential(*modules)
        else:
            resnet = resnet34_encoderdecoder()
            modules = list(resnet.children())[:8]
            self.resnet = nn.Sequential(*modules)

        self.lin1 = nn.Sequential(
            nn.Linear(54, 32),
            nn.ReLU(),
            nn.Dropout(0.5, inplace=False))

        self.lin2  = nn.Sequential(
            nn.Linear(512*32, 8192),
            nn.ReLU(),
            nn.Dropout(0.5, inplace=False))

        self.classifier = nn.Sequential(
            nn.Linear(8192, self.feature_size*self.feature_size*5*self.num_bboxes),
            nn.Sigmoid())

    def forward(self, images):
        batch_size = images.shape[0]

        data = []
        for i in range(6):
            img_warp = kornia.warp_perspective(images[:,i,:,:,:], self.M_matrices[i].unsqueeze(0).repeat(batch_size, 1,1), dsize=(204, 306))
            img_warp = kornia.center_crop(img_warp, (192,288))
            out = self.resnet(img_warp)
            data.append(out.unsqueeze(0))

        agg = torch.cat(data, dim=0)
        agg = torch.max(agg,dim=0)[0]
        agg = agg.view(agg.size(0), 512, -1)
        agg = self.lin1(agg)

        boxes = agg.view(agg.size(0), -1)
        boxes = self.lin2(boxes)
        boxes = self.classifier(boxes)
        boxes = boxes.view(-1, self.feature_size, self.feature_size, 5 * self.num_bboxes)

        return boxes

# Version 3
class Yo3o(nn.Module):
    '''DarkNet Backbone + Mean Aggregation'''
    def __init__(self, feature_size = 20, num_bboxes = 2, device = 'cuda'):
        super().__init__()

        self.M_matrices = torch.tensor([
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
           ]).to(device)

        # Reshape input to from 219x306 to 204x306 using warp
        # Reshape from 204x306 to 288 x 192 using crop
        self.feature_size = feature_size
        self.num_bboxes = num_bboxes

        self.darknet = DarkNet()

        self.lin1 = nn.Sequential(
            nn.Linear(216, 16),
            nn.ReLU(),
            nn.Dropout(0.5, inplace=False))

        self.lin2  = nn.Sequential(
            nn.Linear(16384, 8192),
            nn.ReLU(),
            nn.Dropout(0.5, inplace=False))

        self.classifier = nn.Sequential(
            nn.Linear(8192, self.feature_size*self.feature_size*5*self.num_bboxes),
            nn.Sigmoid())

    def forward(self, images):
        batch_size = images.shape[0]

        data = []
        for i in range(6):
            img_warp = kornia.warp_perspective(images[:,1,:,:,:], self.M_matrices[i].unsqueeze(0).repeat(batch_size, 1,1), dsize=(204, 306))
            img_warp = kornia.center_crop(img_warp, (192,288))
            out = self.darknet(img_warp)
            data.append(out.unsqueeze(1))

        agg = torch.cat(data, dim=1)
        agg = agg.view(agg.size(0), agg.size(1), 1024, -1)
        agg = self.lin1(agg)

        boxes = agg.view(agg.size(0), agg.size(1), -1)
        boxes = torch.mean(boxes,dim=1).squeeze(1)
        boxes = self.lin2(boxes)

        boxes = self.classifier(boxes)
        boxes = boxes.view(-1, self.feature_size, self.feature_size, 5 * self.num_bboxes)

        return boxes

# Version 4
class Yo4o(nn.Module):
    '''ResNet Backbone + Mean Aggregation'''
    def __init__(self, feature_size = 20, num_bboxes = 2, device = 'cuda', load_pretrained = False):
        super().__init__()

        self.feature_size = feature_size
        self.num_bboxes = num_bboxes
        self.device = device

        self.M_matrices = torch.tensor([
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
           ]).to(self.device)

        if load_pretrained:
            resnet = torch.load('../self-sup/resnet34_pretrain')
            modules = list(resnet.children())[:8]
            self.resnet = nn.Sequential(*modules)
        else:
            resnet = resnet34_encoderdecoder()
            modules = list(resnet.children())[:8]
            self.resnet = nn.Sequential(*modules)

        self.lin1 = nn.Sequential(
            nn.Linear(54, 32),
            nn.ReLU(),
            nn.Dropout(0.5, inplace=False))

        self.lin2  = nn.Sequential(
            nn.Linear(512*32, 8192),
            nn.ReLU(),
            nn.Dropout(0.5, inplace=False))

        self.classifier = nn.Sequential(
            nn.Linear(8192, self.feature_size*self.feature_size*5*self.num_bboxes),
            nn.Sigmoid())

    def forward(self, images):
        batch_size = images.shape[0]

        data = []
        for i in range(6):
            img_warp = kornia.warp_perspective(images[:,i,:,:,:], self.M_matrices[i].unsqueeze(0).repeat(batch_size, 1,1), dsize=(204, 306))
            img_warp = kornia.center_crop(img_warp, (192,288))
            out = self.resnet(img_warp)
            data.append(out.unsqueeze(1))

        agg = torch.cat(data, dim=1)
        agg = agg.view(agg.size(0), agg.size(1), 512, -1)
        agg = self.lin1(agg)

        boxes = agg.view(agg.size(0), agg.size(1), -1)
        boxes = torch.mean(boxes,dim=1).squeeze(1)
        boxes = self.lin2(boxes)

        boxes = self.classifier(boxes)
        boxes = boxes.view(-1, self.feature_size, self.feature_size, 5 * self.num_bboxes)

        return boxes

class DarkNet(nn.Module):
    '''DarkNet19 Implimentation'''
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 128, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, images):

        out = self.conv1(images)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        return out


# Resnet code taken from https://github.com/pytorch/vision/blob/v0.2.0/torchvision/models/resnet.py
def bilinear(mode = 'bilinear', scale_factor=2):
    "bilinear upsampling"
    return nn.Upsample(scale_factor=scale_factor, mode=mode)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, mask = None):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, average_pool_size = 7, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(average_pool_size)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class ResNet_EncoderDecoder(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet_EncoderDecoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)


        self.deconv1 = nn.ConvTranspose2d(512*block.expansion, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_d1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_d2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_d3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_d4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_d5 = nn.BatchNorm2d(32)
        ### output size: 64x256x256; apply regressor
        self.classifier = nn.Conv2d(32, 3, kernel_size=3, padding=1, bias=True)
        self.tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def encode(self, x):
        x = self.conv1(x)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def decode(self, x):

        x = self.deconv1(x)
        x = self.bn_d1(x)
        x = self.relu(x)

        x = self.deconv2(x)
        x = self.bn_d2(x)
        x = self.relu(x)

        x = self.deconv3(x)
        x = self.bn_d3(x)
        x = self.relu(x)

        x = self.deconv4(x)
        x = self.bn_d4(x)
        x = self.relu(x)

        x = self.deconv5(x)
        x = self.bn_d5(x)
        x = self.relu(x)

        x = self.classifier(x)
        x = self.tanh(x)

        return x

    def forward(self, x):
        e = self.encode(x)
        d = self.decode(e)
        return d

def resnet34_encoderdecoder(**kwargs):
    """Constructs a ResNet-34 encoder + decoder model.
    """
    model = ResNet_EncoderDecoder(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import kornia
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models

class DarkNet(nn.Module):
    def __init__(self, device):
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

class Yolo(nn.Module):
    def __init__(self, feature_size, num_bboxes, device):
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
        self.conv = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
        )
        self.pool1 = nn.MaxPool1d(54)
        self.lin1 = nn.Sequential(
            nn.Linear(6 * 1024, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5, inplace=False)
        )
        self.lin2 = nn.Sequential(
            nn.Linear(4096, self.feature_size*self.feature_size*5*self.num_bboxes),
            nn.Sigmoid()
        )

    def forward(self, images):
        data = []

        batch_size = images.shape[0]

        for i in range(6):
            img_warp = kornia.warp_perspective(images[:,i,:,:,:], self.M_matrices[i].unsqueeze(0).repeat(batch_size, 1,1), dsize=(204, 306))
            img_warp = kornia.center_crop(img_warp, (192,288))
            out = self.darknet(img_warp)
            out = self.conv(out)
            out = out.view(batch_size,1024,-1)
            out = self.pool1(out).squeeze(-1)
            data.append(out.unsqueeze(1))
        data = torch.cat(data, dim=1)

        data = data.view(out.size(0), -1)
        data = self.lin1(data)
        data = self.lin2(data)
        data = data.view(-1, self.feature_size, self.feature_size, 5 * self.num_bboxes)

        return data


class Yo2o(nn.Module):
    def __init__(self, feature_size = 20, num_bboxes = 2, device = 'cuda'):
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

        resnet = models.resnet34(pretrained=False)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        self.lin1 = nn.Sequential(
            nn.Linear(80, 64),
            nn.ReLU(),
            nn.Dropout(0.5, inplace=False))

        self.lin2  = nn.Sequential(
            nn.Linear(512*64, 8192),
            nn.ReLU(),
            nn.Dropout(0.5, inplace=False))

        self.classifier = nn.Sequential(
            nn.Linear(8192, self.feature_size*self.feature_size*5*self.num_bboxes),
            nn.Sigmoid())

    def forward(self, images):
        batch_size = images.shape[0]

        data = []
        for i in range(batch_size):
            img_warp = kornia.warp_perspective(images[:,i,:,:,:], self.M_matrices[i].unsqueeze(0).repeat(batch_size, 1,1), dsize=(204, 306))
            img_warp = kornia.center_crop(img_warp, (192,288))
            out = self.resnet(img_warp)
            data.append(out.unsqueeze(0))

        agg = torch.cat(data, dim=0)
        agg = torch.max(agg,dim=0)[0]
        agg = agg.view(agg.size(0), 2048, -1)
        agg = self.lin1(agg)

        boxes = agg.view(agg.size(0), -1)
        boxes = self.lin2(boxes)
        boxes = self.classifier(boxes)
        boxes = boxes.view(-1, self.feature_size, self.feature_size, 5 * self.num_bboxes)

        return boxes

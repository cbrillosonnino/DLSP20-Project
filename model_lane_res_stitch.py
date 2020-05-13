
import math
import copy
import torch
import torch.nn as nn
from torch.autograd import Variable
import kornia
import kornia.augmentation as K

#src:https://github.com/suriyasingh/Self-supervision-for-segmenting-overhead-imagery/blob/c232438132cdbdd6a0bd68431c74ea52deb94240/models.py

class Stitch_Classfier(nn.Module):
    def __init__(self, original_model, n_class=2, layers_to_remove=['classifier', 'tanh']):
        super(Stitch_Classfier, self).__init__()
        torch.cuda.manual_seed(7)
        torch.manual_seed(7)
        for layers_ in layers_to_remove:        
            del(original_model._modules[layers_])
                
        self.features = copy.deepcopy(original_model)
        self.d1 = nn.Conv2d(512, n_class, 1)
        self.d2 = nn.Conv2d(256, n_class, 1)
        self.d3 = nn.Conv2d(128, n_class, 1)
        self.d4 = nn.Conv2d(64, n_class, 1)
        self.d5 = nn.Conv2d(32, n_class, 1)

        self.up_maps = nn.ModuleList([nn.UpsamplingBilinear2d((800,800)) for i in range(5)])  #5 scales bilinear maps

        ### initialize new layers with random weights
        for m in [self.d1,self.d2,self.d3,self.d4,self.d5]:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            
    def forward(self, x):
        #encoder: pretrain resnet
        x = self.features.conv1(x)
        x = self.features.bn1(x)
        x = self.features.relu(x)
        x = self.features.maxpool(x)

        layer1 = self.features.layer1(x)
        layer2 = self.features.layer2(layer1) #downsample 2^3
        layer3 = self.features.layer3(layer2) #downsample 2^4
        layer4 = self.features.layer4(layer3) #downsample 2^5

        #decoding + classifier step
        d1 = self.features.relu(self.features.bn_d1(self.features.deconv1(layer4))) 
        score_d1 = self.up_maps[0](self.d1(d1))
        d1 = self.features.relu(self.features.bn_d2(self.features.deconv2(d1)))
        score_d2 = self.up_maps[1](self.d2(d1))
        d1 = self.features.relu(self.features.bn_d3(self.features.deconv3(d1)))
        score_d3 = self.up_maps[2](self.d3(d1))
        d1 = self.features.relu(self.features.bn_d4(self.features.deconv4(d1)))
        score_d4 = self.up_maps[3](self.d4(d1))
        d1 = self.features.relu(self.features.bn_d5(self.features.deconv5(d1)))
        score_d5 = self.up_maps[4](self.d5(d1))

        return score_d1 + score_d2 + score_d3 + score_d4 + score_d5

#serve as the original model. Also used in pretext/transfer learning
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
##helpers

def resnet18_encoderdecoder(**kwargs):
    """Constructs a ResNet-50 encoder + decoder model.
    """
    model = ResNet_EncoderDecoder(BasicBlock, [2, 2, 2, 2], **kwargs)
    
    return model   
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
                     
def stitch(x, M_matrices,M_rotations, M_flip, label=True):
    #Preprocessing: image stitch
    data = [] #list to store all the features maps from multi-views
    for i in range(6):
        #get a batch of *same* view images
        img_batch = x[:,i,:,:,:] # torch.stack(x)[:,i,:,:,:] #
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

    #flip 90 degree
    agg = kornia.warp_affine(agg, M_flip.repeat(len(x), 1,1), dsize=(438,438))
    #Normalize color
    if label:
        normalize = K.Normalize(torch.tensor([0.698, 0.718, 0.730]),
                              torch.tensor([0.322, 0.313, 0.308]))
    else:
        normalize = K.Normalize(torch.tensor([0.548, 0.597, 0.630]),
                         torch.tensor([0.339, 0.340, 0.342]))

    return normalize(agg)

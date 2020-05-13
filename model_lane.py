
import math
import copy
import torch
import torch.nn as nn
from torch.autograd import Variable
import kornia

#src:https://github.com/suriyasingh/Self-supervision-for-segmenting-overhead-imagery/blob/c232438132cdbdd6a0bd68431c74ea52deb94240/models.py

class Multi_Classfier(nn.Module):
    def __init__(self, original_model, n_class=21, layers_to_remove=['classifier', 'tanh']):
        super(Multi_Classfier, self).__init__()
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
        self.proj = nn.Linear(2048*6, 2048, bias=False) #might blow up

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

            
    def forward(self, x6):
    #in our case: x is BEV_inputs: size [6,bs,3,h,w]      
        agg = [] #store each view resnet encoder output
        for i in range(6):
            x = self.features.conv1(x6[i])
            x = self.features.bn1(x)
            x = self.features.relu(x)
            x = self.features.maxpool(x)
            layer1 = self.features.layer1(x)
            layer2 = self.features.layer2(layer1) #downsample 2^3
            layer3 = self.features.layer3(layer2) #downsample 2^4
            layer4 = self.features.layer4(layer3) #downsample 2^5
            agg.append(layer4.unsqueeze(0))
        agg = torch.cat(agg,dim=0)
        agg = self.proj(agg.permute(1,3,4,2,0).reshape(len(x), 6,9, 2048*6))  #first reshpe to [bs,h,w,2048,6] then squeeze to [bs,h,w,2048*6]
        agg = agg.permute(0,3,1,2) #shape[bs, 2048, 6, 9]

        #decoding + classifier step
        d1 = self.features.relu(self.features.bn_d1(self.features.deconv1(agg))) #take in aggregated!
        score_d1 = self.up_maps[0](self.d1(d1))
        d1 = self.features.relu(self.features.bn_d2(self.features.deconv2(d1)))
        score_d2 = self.up_maps[1](self.d2(d1))
        d1 = self.features.relu(self.features.bn_d3(self.features.deconv3(d1)))
        score_d3 = self.up_maps[2](self.d3(d1))
        d1 = self.features.relu(self.features.bn_d4(self.features.deconv4(d1)))
        score_d4 = self.up_maps[3](self.d4(d1))
        d1 = self.features.relu(self.features.bn_d5(self.features.deconv5(d1)))
        score_d5 = self.up_maps[4](self.d5(d1))
        #print(score_d1.shape, score_d2.shape, score_d3.shape, score_d4.shape, score_d5.shape)

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
##helpers

def bilinear(mode = 'bilinear', scale_factor=2):
    "bilinear upsampling"
    return nn.Upsample(scale_factor=scale_factor, mode=mode)

def crop(variable,tr,tc): #Might upsample greater than original size
    r, c = variable.size()[-2:]
    r1 = int(round((r - tr) / 2.))
    c1 = int(round((c - tc) / 2.))
    return variable[:,:, r1:r1+tr,c1:c1+tc]

def resnet50_encoderdecoder(**kwargs):
    """Constructs a ResNet-50 encoder + decoder model.
    """
    model = ResNet_EncoderDecoder(Bottleneck, [3, 4, 6, 3], **kwargs)
    
    return model

def warp_transform(imgs, M_matrices):
  '''
  input: tuple of tensor, each element is [6,3,256,306]
  for eval: imgs is one element only
  '''
  data = []
  for i in range(6): #loop through each view
    img_batch = imgs[:,i,:,:,:].cuda()
    img_batch = kornia.warp_perspective(img_batch, M_matrices[i], dsize=(204, 306))
    img_batch = kornia.center_crop(img_batch, (192,288))
    data.append(img_batch.unsqueeze(0))
  return torch.cat(data,dim=0)

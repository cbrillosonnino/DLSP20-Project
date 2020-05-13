"""
You need to implement all four functions in this file and also put your team info as a variable
Then you should submit the python file with your model class, the state_dict, and this file
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms


###import model class module
from model_lane_res_stitch import Stitch_Classfier, resnet18_encoderdecoder, stitch
from model_bb_stitch import Yo4o_stitch

###utilities
import util
from loss import Loss
import kornia

# Put your transform function here, we will use it for our dataloader
# For bounding boxes task
def get_transform_task1(): 
    return transforms.Compose([
                transforms.ToTensor()
                # transforms.Normalize((0.698, 0.718, 0.730),
                #                      (0.322, 0.313, 0.308))
                ])
# For road map task
def get_transform_task2(): 
    return transforms.Compose([
                transforms.ToTensor()
                # transforms.Normalize((0.698, 0.718, 0.730),
                #                      (0.322, 0.313, 0.308))
                ])

class ModelLoader():
    # Fill the information for your team
    team_name = 'DreamTeam'
    team_number = 2
    round_number = 2
    team_member = ['Teresa Ningyuan Huang','Charles Brillo-Sonnino']
    contact_email = 'nh1724@nyu.edu', 'cbs488@nyu.edu'

    def __init__(self, model_file='best_lane_res18_stitch'):
        net = resnet18_encoderdecoder().cuda()
        self.model = Stitch_Classfier(net, n_class = 2).cuda()
        self.model.load_state_dict(torch.load(model_file))
        self.model.eval()
        
        self.bb_model = Yo4o_stitch(20, 2).cuda()
        checkpoint = torch.load('stitch_bbox_best.pth.tar')
        self.bb_model.load_state_dict(checkpoint['model'])
        self.loss = Loss(20, 2)
        self.bb_model.eval()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        

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

        # rotation matrices
        self.M_rotations = torch.tensor([
        [[ 5.0000e-01,  8.6603e-01, -1.8330e+01],
          [-8.6603e-01,  5.0000e-01,  1.8725e+02]],

        [[ 1.0000e+00,  0.0000e+00,  0.0000e+00],
          [-0.0000e+00,  1.0000e+00,  0.0000e+00]],

        [[ 5.0000e-01, -8.6603e-01,  1.7133e+02],
          [ 8.6603e-01,  5.0000e-01, -7.7752e+01]],

        [[-5.0000e-01,  8.6603e-01,  1.3467e+02],
          [-8.6603e-01, -5.0000e-01,  2.9675e+02]],

        [[-1.0000e+00,  8.7423e-08,  3.0600e+02],
          [-8.7423e-08, -1.0000e+00,  2.1900e+02]],

        [[-5.0000e-01, -8.6603e-01,  3.2433e+02],
          [ 8.6603e-01, -5.0000e-01,  3.1748e+01]]]).to(self.device)

        #flip 90 degree to align car facing right
        self.M_flip = torch.tensor([
        [[-4.3711e-08, -1.0000e+00,  4.3800e+02],
        [ 1.0000e+00, -4.3711e-08,  0.0000e+00]]]).to(self.device)
        

    def get_bounding_boxes(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object
        with torch.no_grad():
          output = self.bb_model(samples.cuda())
          boxes = []
          batch_size = output.shape[0]
          for i in range(batch_size):
            hypothesis = output[i]
            target, _ = self.loss.decode(hypothesis, conf_thresh = 0.5)
            boxes.append(target)
        return tuple(boxes)

    def get_binary_road_map(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800] 
        BEV_inputs = stitch(samples.cuda(),self.M_matrices, self.M_rotations, self.M_flip, label=True)
        pred = self.model(BEV_inputs)
        return pred.data.max(1)[1]  
 

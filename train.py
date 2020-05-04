import argparse, os
from pathlib import Path
import shutil

import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import kornia
import torch.nn.functional as F
from torch.autograd import Variable

from data_loading import get_loaders


def get_parser():
    parser = argparse.ArgumentParser(description='Flickr30k Training')
    parser.add_argument('-batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('-epoch', type=int, default=30,
                        help='number of training epochs')
    parser.add_argument('-lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('-feature_size', type=int, default=80,
                        help='image grid dimensions')
    parser.add_argument('-num_bboxes', type=int, default=1,
                        help='number of boxes per grid cell')
    parser.add_argument('--save', type=str, default=Path.cwd(),
                        help='directory to save logs and models.')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    return parser

def main():

    cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = get_parser().parse_args()

    learning_rate = args.lr
    start_epoch = 0
    min_val_loss = 1_000_000

    trainloader, valloader = get_loaders('labeled', batch_size = args.batch_size)

    model = Yolo(args.feature_size, args.num_bboxes).to(device)
    loss_fxn = Loss(args.feature_size, args.num_bboxes)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            max_BLEU = checkpoint['max_BLEU']
            encoder.load_state_dict(checkpoint['encoder'])
            decoder.load_state_dict(checkpoint['decoder'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else: print("No checkpoint found at '{}'".format(args.resume))

    if not args.resume:
        file = open(f'{args.save}/resuts.txt','a')
        file.write('Train_Loss,Val_Loss \n')
        file.close()

    train_losses = AverageMeter()
    val_losses = AverageMeter()
    for epoch in range(args.epoch):
        print('Epoch {}'.format(epoch+1))
        model.train()
        print('training...')
        train_losses.reset()
        val_losses.reset()
        for i, (sample, target, road_image) in enumerate(trainloader):
            if i%100 ==  0:
                print('[{}/{}] {}'.format(i,len(trainloader), train_losses.avg))
            batch_size = len(sample)
            img_batch = torch.stack(sample).to(device)
            output = model(img_batch)
            loss = loss_fxn(output, target)
            train_losses.update(loss, batch_size)
            model.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        print('validating...')
        for i, (sample, target, road_image) in enumerate(valloader):
            with torch.no_grad():
                batch_size = len(sample)
                img_batch = torch.stack(sample).to(device)
                output = model(img_batch)
                loss = loss_fxn(output, target)
                val_losses.update(loss, batch_size)

        if epoch+1 % 10 == 0:
            learning_rate /= 10
            for param_group in optimizer.param_groups: param_group['lr'] = learning_rate

        val_loss = val_losses.avg
        print('Validation Loss = {}'.format(val_loss))
        is_best = val_loss < min_val_loss
        max_BLEU = min(val_loss, min_val_loss)

        file = open(f'{args.save}/resuts.txt','a')
        file.write('{},{}\n'.format(train_losses.avg,val_loss))
        file.close()
        
        save_checkpoint({
            'epoch': epoch + 1, 'model': model.state_dict(),
            'min_val_loss': min_val_loss, 'optimizer' : optimizer.state_dict(),
        }, is_best, args.save)


class DarkNet(nn.Module):
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

        self.conv6 = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
        )

    def forward(self, images):

        out = self.conv1(images)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)

        return out

class Yolo(nn.Module):
    def __init__(self, feature_size, num_bboxes):
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

        # Reshape input to from 219x305 to 204x306 using warp
        # Reshape from 204x306 to 288 x 192 using crop
        self.feature_size = feature_size
        self.num_bboxes = num_bboxes

        self.darknet = DarkNet()
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
            img_warp = kornia.warp_perspective(img_batch[:,i,:,:,:], self.M_matrices[i].unsqueeze(0).repeat(len(images), 1,1), dsize=(204, 306))
            img_warp = kornia.center_crop(img_warp, (192,288))
            out = self.darknet(img_warp)
            out = out.view(batch_size,1024,-1)
            out = self.pool1(out).squeeze(-1)
            data.append(out.unsqueeze(1))
        data = torch.cat(data, dim=1)

        data = data.view(out.size(0), -1)
        data = self.lin1(data)
        data = self.lin2(data)
        data = data.view(-1, self.feature_size, self.feature_size, 5 * self.num_bboxes)

        return data

class Loss(nn.Module):

    def __init__(self, feature_size=80, num_bboxes=1, lambda_coord=5.0, lambda_noobj=0.5):
        """ Constructor.
        Args:
            feature_size: (int) size of input feature map.
            num_bboxes: (int) number of bboxes per each cell.
            lambda_coord: (float) weight for bbox location/size losses.
            lambda_noobj: (float) weight for no-objectness loss.
        """
        super(Loss, self).__init__()

        self.S = feature_size
        self.B = num_bboxes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj


    def compute_iou(self, bbox1, bbox2):
        """ Compute the IoU (Intersection over Union) of two set of bboxes, each bbox format: [x1, y1, x2, y2].
        Args:
            bbox1: (Tensor) bounding bboxes, sized [N, 4].
            bbox2: (Tensor) bounding bboxes, sized [M, 4].
        Returns:
            (Tensor) IoU, sized [N, M].
        """
        N = bbox1.size(0)
        M = bbox2.size(0)

        # Compute left-top coordinate of the intersections
        lt = torch.max(
            bbox1[:, :2].unsqueeze(1).expand(N, M, 2), # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bbox2[:, :2].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        # Conpute right-bottom coordinate of the intersections
        rb = torch.min(
            bbox1[:, 2:].unsqueeze(1).expand(N, M, 2), # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bbox2[:, 2:].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        # Compute area of the intersections from the coordinates
        wh = rb - lt   # width and height of the intersection, [N, M, 2]
        wh[wh < 0] = 0 # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1] # [N, M]

        # Compute area of the bboxes
        area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1]) # [N, ]
        area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1]) # [M, ]
        area1 = area1.unsqueeze(1).expand_as(inter) # [N, ] -> [N, 1] -> [N, M]
        area2 = area2.unsqueeze(0).expand_as(inter) # [M, ] -> [1, M] -> [N, M]

        # Compute IoU from the areas
        union = area1 + area2 - inter # [N, M, 2]
        iou = inter / union           # [N, M, 2]

        return iou

    def return_target_tensor(self, target):
        tensor_list = []
        B = self.B
        for item in target:
            boxes = item['bounding_box']
            boxes[:,0] = (boxes[:,0]+40)/80
            boxes[:,1] = torch.abs(boxes[:,1]-40)/80

            boxes_max_x = boxes[:, 0].max(dim=1)[0]
            boxes_min_x = boxes[:, 0].min(dim=1)[0]
            boxes_max_y = boxes[:, 1].max(dim=1)[0]
            boxes_min_y = boxes[:, 1].min(dim=1)[0]

            cell_size = 1.0 / 80.0
            targ = torch.zeros(80, 80, B*5)

            boxes = torch.stack((boxes_min_x, boxes_min_y, boxes_max_x, boxes_max_y), 1)

            boxes_wh = boxes[:, 2:] - boxes[:, :2] # width and height for each box, [n, 2]
            boxes_xy = (boxes[:, 2:] + boxes[:, :2]) / 2.0 # center x & y for each box, [n, 2]

            for b in range(boxes.size(0)):
                xy, wh = boxes_xy[b], boxes_wh[b]

                ij = (xy / cell_size).ceil() - 1.0
                i, j = int(ij[0]), int(ij[1]) # y & x index which represents its location on the grid.
                x0y0 = ij * cell_size # x & y of the cell left-top corner.
                xy_normalized = (xy - x0y0) / cell_size

                for k in range(B):
                    s = 5 * k
                    targ[j, i, s  :s+2] = xy_normalized
                    targ[j, i, s+2:s+4] = wh
                    targ[j, i, s+4    ] = 1.0
            tensor_list.append(targ)
        return torch.stack(tensor_list)

    def forward(self, pred_tensor, target):
        """ Compute loss for YOLO training.
        Args:
            pred_tensor: (Tensor) predictions, sized [n_batch, S, S, Bx5+C], 5=len([x, y, w, h, conf]).
            target_tensor: (Tensor) targets, sized [n_batch, S, S, Bx5+C].
        Returns:
            (Tensor): loss, sized [1, ].
        """
        # TODO: Romove redundant dimensions for some Tensors.

        target_tensor = self.return_target_tensor(target).cuda()

        S, B = self.S, self.B
        N = 5 * B# 5=len([x, y, w, h, conf]

        batch_size = pred_tensor.size(0)
        coord_mask = target_tensor[:, :, :, 4] > 0  # mask for the cells which contain objects. [n_batch, S, S]
        noobj_mask = target_tensor[:, :, :, 4] == 0 # mask for the cells which do not contain objects. [n_batch, S, S]
        coord_mask = coord_mask.unsqueeze(-1).expand_as(target_tensor) # [n_batch, S, S] -> [n_batch, S, S, N]
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(target_tensor) # [n_batch, S, S] -> [n_batch, S, S, N]

        coord_pred = pred_tensor[coord_mask == 1].view(-1, N)            # pred tensor on the cells which contain objects. [n_coord, N]
                                                                    # n_coord: number of the cells which contain objects.
        bbox_pred = coord_pred[:, :5*B].contiguous().view(-1, 5)    # [n_coord x B, 5=len([x, y, w, h, conf])]                        # [n_coord, C]

        coord_target = target_tensor[coord_mask == 1].view(-1, N)        # target tensor on the cells which contain objects. [n_coord, N]
                                                                    # n_coord: number of the cells which contain objects.
        bbox_target = coord_target[:, :5*B].contiguous().view(-1, 5)# [n_coord x B, 5=len([x, y, w, h, conf])]

        # Compute loss for the cells with no object bbox.
        noobj_pred = pred_tensor[noobj_mask].view(-1, N)        # pred tensor on the cells which do not contain objects. [n_noobj, N]
                                                                # n_noobj: number of the cells which do not contain objects.
        noobj_target = target_tensor[noobj_mask].view(-1, N)    # target tensor on the cells which do not contain objects. [n_noobj, N]
                                                                # n_noobj: number of the cells which do not contain objects.
        noobj_conf_mask = torch.cuda.ByteTensor(noobj_pred.size()).fill_(0) # [n_noobj, N]
        for b in range(B):
            noobj_conf_mask[:, 4 + b*5] = 1 # noobj_conf_mask[:, 4] = 1; noobj_conf_mask[:, 9] = 1
        noobj_pred_conf = noobj_pred[noobj_conf_mask == 1]       # [n_noobj, 2=len([conf1, conf2])]
        noobj_target_conf = noobj_target[noobj_conf_mask == 1]   # [n_noobj, 2=len([conf1, conf2])]
        loss_noobj = F.mse_loss(noobj_pred_conf, noobj_target_conf, reduction='sum')

        # Compute loss for the cells with objects.
        coord_response_mask = torch.cuda.ByteTensor(bbox_target.size()).fill_(0)    # [n_coord x B, 5]
        coord_not_response_mask = torch.cuda.ByteTensor(bbox_target.size()).fill_(1)# [n_coord x B, 5]
        bbox_target_iou = torch.zeros(bbox_target.size()).cuda()                    # [n_coord x B, 5], only the last 1=(conf,) is used

        # Choose the predicted bbox having the highest IoU for each target bbox.
        for i in range(0, bbox_target.size(0), B):
            pred = bbox_pred[i:i+B] # predicted bboxes at i-th cell, [B, 5=len([x, y, w, h, conf])]
            pred_xyxy = Variable(torch.FloatTensor(pred.size())) # [B, 5=len([x1, y1, x2, y2, conf])]
            # Because (center_x,center_y)=pred[:, 2] and (w,h)=pred[:,2:4] are normalized for cell-size and image-size respectively,
            # rescale (center_x,center_y) for the image-size to compute IoU correctly.
            pred_xyxy[:,  :2] = pred[:, 2]/float(S) - 0.5 * pred[:, 2:4]
            pred_xyxy[:, 2:4] = pred[:, 2]/float(S) + 0.5 * pred[:, 2:4]

            target = bbox_target[i] # target bbox at i-th cell. Because target boxes contained by each cell are identical in current implementation, enough to extract the first one.
            target = bbox_target[i].view(-1, 5) # target bbox at i-th cell, [1, 5=len([x, y, w, h, conf])]
            target_xyxy = Variable(torch.FloatTensor(target.size())) # [1, 5=len([x1, y1, x2, y2, conf])]
            # Because (center_x,center_y)=target[:, 2] and (w,h)=target[:,2:4] are normalized for cell-size and image-size respectively,
            # rescale (center_x,center_y) for the image-size to compute IoU correctly.
            target_xyxy[:,  :2] = target[:, 2]/float(S) - 0.5 * target[:, 2:4]
            target_xyxy[:, 2:4] = target[:, 2]/float(S) + 0.5 * target[:, 2:4]

            iou = self.compute_iou(pred_xyxy[:, :4], target_xyxy[:, :4]) # [B, 1]
            max_iou, max_index = iou.max(0)
            max_index = max_index.data.cuda()

            coord_response_mask[i+max_index] = 1
            coord_not_response_mask[i+max_index] = 0

            # "we want the confidence score to equal the intersection over union (IOU) between the predicted box and the ground truth"
            # from the original paper of YOLO.
            bbox_target_iou[i+max_index, torch.LongTensor([4]).cuda()] = (max_iou).data.cuda()
        bbox_target_iou = Variable(bbox_target_iou).cuda()

        # BBox location/size and objectness loss for the response bboxes.
        bbox_pred_response = bbox_pred[coord_response_mask == 1].view(-1, 5)      # [n_response, 5]
        bbox_target_response = bbox_target[coord_response_mask == 1].view(-1, 5)  # [n_response, 5], only the first 4=(x, y, w, h) are used
        target_iou = bbox_target_iou[coord_response_mask == 1].view(-1, 5)        # [n_response, 5], only the last 1=(conf,) is used
        loss_xy = F.mse_loss(bbox_pred_response[:, :2], bbox_target_response[:, :2], reduction='sum')
        loss_wh = F.mse_loss(torch.sqrt(bbox_pred_response[:, 2:4]), torch.sqrt(bbox_target_response[:, 2:4]), reduction='sum')
        loss_obj = F.mse_loss(bbox_pred_response[:, 4], target_iou[:, 4], reduction='sum')

        # Total loss
        loss = self.lambda_coord * (loss_xy + loss_wh) + loss_obj + self.lambda_noobj * loss_noobj
        loss = loss / float(batch_size)

        return loss


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, save, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, f'{save}/model_best.pth.tar')


if __name__ == '__main__': main()

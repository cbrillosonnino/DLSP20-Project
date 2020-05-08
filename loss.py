import torch
import torch.nn as nn
import torch.nn.functional as F

from helper import compute_ats_bounding_boxes

class Loss(nn.Module):

    def __init__(self, feature_size=20, num_bboxes=2, lambda_coord=5.0, lambda_noobj=0.5):
        """ Constructor.
        Args:
            feature_size: (int) size of input feature map.
            num_bboxes: (int) number of bboxes per each cell.
            lambda_coord: (float) weight for bbox location/size losses.
            lambda_noobj: (float) weight for no-objectness loss.
        """
        super(Loss, self).__init__()

        self.feature_size = feature_size
        self.num_bboxes = num_bboxes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def return_target_tensor(self, inputs):
        """encode box coordinates into tensor"""
        tensor_list = []

        for item in inputs:
            boxes = item['bounding_box'].clone()
            boxes[:,0] = (boxes[:,0]+40)/80
            boxes[:,1] = torch.abs(boxes[:,1]-40)/80

            boxes_max_x = boxes[:, 0].max(dim=1)[0]
            boxes_min_x = boxes[:, 0].min(dim=1)[0]
            boxes_max_y = boxes[:, 1].max(dim=1)[0]
            boxes_min_y = boxes[:, 1].min(dim=1)[0]

            cell_size = 1.0 / self.feature_size
            target_tensor = torch.zeros(self.feature_size, self.feature_size, self.num_bboxes*5)

            boxes_wh = torch.stack((boxes_max_x-boxes_min_x, boxes_max_y-boxes_min_y), 1)
            boxes_xy = torch.stack(((boxes_max_x+boxes_min_x), (boxes_max_y+boxes_min_y)), 1)/2


#             boxes = torch.stack((boxes_min_x, boxes_min_y, boxes_max_x, boxes_max_y), 1)

#             boxes_wh = boxes[:, 2:] - boxes[:, :2] # width and height for each box, [n, 2]
#             boxes_xy = (boxes[:, 2:] + boxes[:, :2]) / 2.0 # center x & y for each box, [n, 2]

            for b in range(boxes.size(0)):
                xy, wh = boxes_xy[b], boxes_wh[b]

                ij = (xy / cell_size).ceil() - 1.0
                i, j = int(ij[0]), int(ij[1]) # y & x index which represents its location on the grid.
                x0y0 = ij * cell_size # x & y of the cell left-top corner.
                xy_normalized = (xy - x0y0) / cell_size
                # only works for num_bboxes=2
                if target_tensor[j,i,4] == 0:
                    for k in range(2):
                        s = 5 * k
                        target_tensor[j, i, s  :s+2] = xy_normalized
                        target_tensor[j, i, s+2:s+4] = wh
                        target_tensor[j, i, s+4    ] = 1.0
#                     s=0
#                     target_tensor[j, i, s  :s+2] = xy_normalized
#                     target_tensor[j, i, s+2:s+4] = wh
#                     target_tensor[j, i, s+4    ] = 1.0
                else:
                    s = 5
                    target_tensor[j, i, s  :s+2] = xy_normalized
                    target_tensor[j, i, s+2:s+4] = wh
                    target_tensor[j, i, s+4    ] = 1.0

            tensor_list.append(target_tensor)
        return torch.stack(tensor_list)

    def decode(self, pred_tensor, conf_thresh = 0.5, decice = 'cuda'):
        """decode single tensor into box coordinates"""

        boxes , confidences = [], []

        cell_size = 1.0 / self.feature_size

        # 1st prediction
        estimates_boxes = pred_tensor[:, :, :5]
        for i in range(self.feature_size): # for x-dimension.
            for j in range(self.feature_size): # for y-dimension.
                prob = estimates_boxes[j, i, 4]
                if prob > conf_thresh:
                    box = pred_tensor[j, i, :4].to(device)
                    x0y0_normalized = torch.FloatTensor([i, j]).to(device) * cell_size
                    xy_normalized = box[:2] * cell_size + x0y0_normalized
                    wh_normalized = box[2:]
                    box_xyxy = torch.FloatTensor(2,4).to(device)
                    box_xyxy[:,0] = xy_normalized - 0.5 * wh_normalized
                    box_xyxy[:,3] = xy_normalized + 0.5 * wh_normalized
                    box_xyxy[0,1] = box_xyxy[0,0]
                    box_xyxy[1,1] = box_xyxy[1,3]
                    box_xyxy[0,2] = box_xyxy[0,3]
                    box_xyxy[1,2] = box_xyxy[1,0]
                    boxes.append(box_xyxy)

        # 2nd prediction
        estimates_boxes = pred_tensor[:, :, 5:10]
        for i in range(self.feature_size): # for x-dimension.
            for j in range(self.feature_size): # for y-dimension.
                prob = estimates_boxes[j, i, 4]
                if prob > conf_thresh:
                    box = pred_tensor[j, i, :4].to(device)
                    x0y0_normalized = torch.FloatTensor([i, j]).to(device) * cell_size
                    xy_normalized = box[:2] * cell_size + x0y0_normalized
                    wh_normalized = box[2:]
                    box_xyxy = torch.FloatTensor(2,4).to(device)
                    box_xyxy[:,0] = xy_normalized - 0.5 * wh_normalized
                    box_xyxy[:,3] = xy_normalized + 0.5 * wh_normalized
                    box_xyxy[0,1] = box_xyxy[0,0]
                    box_xyxy[1,1] = box_xyxy[1,3]
                    box_xyxy[0,2] = box_xyxy[0,3]
                    box_xyxy[1,2] = box_xyxy[1,0]
                    boxes.append(box_xyxy)

        if len(boxes) > 0:
            boxes = torch.stack(boxes, 0)
        else:
            boxes = torch.rand(1,2,4)

        boxes[:,0] = boxes[:,0]*80-40
        boxes[:,1] = -((boxes[:,1]*80)-40)

        return boxes

    def validate(self, predictions, inputs, conf_thresh = 0.5):
        """compute average threat score"""

        ats = 0

        batch_size = predictions.shape[0]

        for element in range(batch_size):
            target = inputs[element]['bounding_box']
            predict = self.decode(predictions[element], conf_thresh = conf_thresh)

            ats += compute_ats_bounding_boxes(target,predict)

        return ats

    def compute_iou(self, bbox1, bbox2):
        """
        Compute the intersection over union of two set of boxes, each box is [x1,y1,w,h]
        :param bbox1: (tensor) bounding boxes, size [N,4]
        :param bbox2: (tensor) bounding boxes, size [M,4]
        :return:
        """
        # compute [x1,y1,x2,y2] w.r.t. top left and bottom right coordinates separately
        b1x1y1 = bbox1[:,:2]-bbox1[:,2:]**2 # [N, (x1,y1)=2]
        b1x2y2 = bbox1[:,:2]+bbox1[:,2:]**2 # [N, (x2,y2)=2]
        b2x1y1 = bbox2[:,:2]-bbox2[:,2:]**2 # [M, (x1,y1)=2]
        b2x2y2 = bbox2[:,:2]+bbox2[:,2:]**2 # [M, (x1,y1)=2]
        box1 = torch.cat((b1x1y1.view(-1,2), b1x2y2.view(-1, 2)), dim=1) # [N,4], 4=[x1,y1,x2,y2]
        box2 = torch.cat((b2x1y1.view(-1,2), b2x2y2.view(-1, 2)), dim=1) # [M,4], 4=[x1,y1,x2,y2]
        N = box1.size(0)
        M = box2.size(0)

        tl = torch.max(
            box1[:,:2].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,:2].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )
        br = torch.min(
            box1[:,2:].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,2:].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = br - tl  # [N,M,2]
        wh[(wh<0).detach()] = 0
        #wh[wh<0] = 0
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
        area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou

    def forward(self, pred_tensor, target, device):
        batch_size = pred_tensor.shape[0]

        target_tensor = self.return_target_tensor(target).to(device)

        target_tensor = target_tensor.view(batch_size,-1,5*self.num_bboxes) # [n_batch, S, S, 5*B] -> [n_batch, S*S, 5*B]
        pred_tensor = pred_tensor.view(batch_size,-1,5*self.num_bboxes)# [n_batch, S, S, 5*B] -> [n_batch, S*S, 5*B]

        coord_mask = target_tensor[:, :, 4] > 0  # mask for the cells which contain objects. [n_batch, S, S]
        noobj_mask = target_tensor[:, :, 4] == 0 # mask for the cells which do not contain objects. [n_batch, S, S]
        coord_mask = coord_mask.unsqueeze(-1).expand_as(target_tensor) # [n_batch, S, S] -> [n_batch, S, S, N]
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(target_tensor) # [n_batch, S, S] -> [n_batch, S, S, N]

        coord_target = target_tensor[coord_mask].view(-1,5*self.num_bboxes)
        coord_pred = pred_tensor[coord_mask].view(-1,5*self.num_bboxes)

        box_pred = coord_pred[:,:self.num_bboxes*5].contiguous().view(-1,5)
        box_target = coord_target[:,:self.num_bboxes*5].contiguous().view(-1,5)

        noobj_target = target_tensor[noobj_mask].view(-1,5*self.num_bboxes)
        noobj_pred = pred_tensor[noobj_mask].view(-1,5*self.num_bboxes)

        # No object loss
        noobj_target_mask = torch.zeros(noobj_target.shape, dtype=torch.bool).to(device)
        for i in range(self.num_bboxes):
            noobj_target_mask[:,i*5+4] = True
        noobj_target_c = noobj_target[noobj_target_mask].to(device)
        noobj_pred_c = noobj_pred[noobj_target_mask].to(device)

        noobj_loss = F.mse_loss(noobj_pred_c, noobj_target_c, reduction='sum')

        # Object loss
        coord_response_mask = torch.zeros(box_target.shape, dtype=torch.bool).to(device)
        coord_not_response_mask = torch.ones(box_target.shape, dtype=torch.bool).to(device)

        for i in range(0,box_target.shape[0],self.num_bboxes):
            box1 = box_pred[i:i+self.num_bboxes].to(device)
            box2 = box_target[i:i+self.num_bboxes].to(device)
            iou = self.compute_iou(box1[:, :4], box2[:, :4])
            max_iou, max_index = iou.max(0)
            coord_response_mask[i+max_index[0]]=True
            coord_not_response_mask[i+max_index[0]]=False


        # 1. response loss
        box_pred_response = box_pred[coord_response_mask].view(-1, 5).to(device)
        box_target_response = box_target[coord_response_mask].view(-1, 5).to(device)
        contain_loss = F.mse_loss(box_pred_response[:, 4], box_target_response[:, 4], reduction='sum')
        position_loss = F.mse_loss(box_pred_response[:, :2], box_target_response[:, :2], reduction='sum')
        dimension_loss = F.mse_loss(torch.sqrt(box_pred_response[:, 2:4]), torch.sqrt(box_target_response[:, 2:4]), reduction='sum')

        # # 2. not response loss
        # box_pred_not_response = box_pred[coord_not_response_mask].view(-1, 5)
        # box_target_not_response = box_target[coord_not_response_mask].view(-1, 5)

        # compute total loss
        total_loss = self.lambda_coord * position_loss +\
                     self.lambda_coord * dimension_loss +\
                     contain_loss +\
                     self.lambda_noobj * noobj_loss

        return total_loss


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

# coding: utf-8
import argparse
import json
from json import dumps
import time
from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
from helper import collate_fn, compute_ts_road_map
from torchvision import transforms

from model_lane import Multi_UNet
from data_loading import get_loaders
import util

#import cv2
import kornia

def main(args):
	# Set up logging and devices
	args.save_dir = util.get_save_dir(args.save_dir, args.name, training=True)
	log = util.get_logger(args.save_dir, args.name)
	tbx = SummaryWriter(args.save_dir)
	device, args.gpu_ids = util.get_available_devices()
	log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
	args.batch_size *= max(1, len(args.gpu_ids))

	# Set random seed
	log.info(f'Using random seed {args.seed}...')
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)

	# Get model
	log.info('Building model...')
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
	M_matrices = M_matrices.cuda()
	print('M on cuda?',M_matrices.is_cuda)

	#load data
	train_loader, val_loader = get_loaders('labeled', batch_size = args.batch_size)
	# initialize model
	model = Multi_UNet(n_channels=3, n_classes=2, BEV=args.use_BEV, Bilinear=args.use_Bilinear)
	#model = nn.DataParallel(model, args.gpu_ids)
	model.to(device)  
	optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4, betas=(0.9, 0.999))
	criterion = nn.CrossEntropyLoss() #binary classification (combined softmax + NLL)

	# Get saver
	saver = util.CheckpointSaver(args.save_dir,
	                         max_checkpoints=args.max_checkpoints,
	                         metric_name='Roadmap Threast Score',
	                         maximize_metric=True, #max threat score
	                         log=log)
	#Training  
	log.info('Training...')
	for epoch in range(args.num_epochs):
		log.info(f'Starting epoch {epoch}...')
		model.train()
		with torch.enable_grad(), \
                tqdm(total=len(train_loader.dataset)) as progress_bar:
			for i, data in enumerate(train_loader):
			    sample, target, road_image = data
			    sample = torch.stack(sample).to(device)
			    road_image = torch.stack(road_image).long().to(device)

			    optimizer.zero_grad()
			    pred = model(sample, M_matrices)
			    loss = criterion(pred,road_image)
			    loss.backward()
			    optimizer.step()
			    # if i % 100 == 0:
			    #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
			    #         epoch, i, len(train_loader),
			    #         100 * i / len(train_loader), loss.item()))
			    # log info
			    batch_size = len(sample)
			    progress_bar.update(batch_size)
			    progress_bar.set_postfix(epoch=epoch, CrossEntropyLoss=loss.item())
			    tbx.add_scalar('train/CrossEntropyLoss', loss.item())
			    tbx.add_scalar('train/LR', optimizer.param_groups[0]['lr'])


		#Validation
		model.eval()
		log.info(f'Evaluating epoch {epoch}...')

		total = 0
		total_ts_road_map = 0
		with torch.no_grad(), \
            tqdm(total=len(val_loader.dataset)) as progress_bar:
		    for i, data in enumerate(val_loader):
		        total += 1
		        sample, target, road_image = data
		        sample = torch.stack(sample).to(device)
		        road_image = torch.stack(road_image).long().to(device)
		        pred = model(sample, M_matrices)
		        predicted_road_map = pred.data.max(1)[1] # get the index of the max log-probability       
		        ts_road_map = compute_ts_road_map(predicted_road_map, road_image)
		        total_ts_road_map += ts_road_map

		        #if opt.verbose:
		        #print(f'{i} - Road Map Score: {ts_road_map:.4}')		
		print(f'Road Map Score: {total_ts_road_map / total:.4}')
		#save model checkpoint: step, model, metric_val, device
		saver.save(epoch, model, total_ts_road_map/total, device)

if __name__ == '__main__':  
    parser = argparse.ArgumentParser(description='Multi-view Unet')
    parser.add_argument('--name',
                        '-n',
                        type=str,
                        required=True,
                        help='Name to identify training or test run.')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='number of training epochs')
    parser.add_argument('--lr',type=float, default=0.01,
                        help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='batch size')
    parser.add_argument('--use_BEV', action='store_true',
                        help='use BEV transformed images')
    parser.add_argument('--use_Bilinear',action='store_true',
                        help='use fixed bilinear upsampling or learned weights')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--save_dir',
                        type=str,
                        default='./save/',
                        help='Base directory for saving information.')
    parser.add_argument('--save', type=str, default='model.pt',
                        help='path to save the final model')
    parser.add_argument('--max_checkpoints',
                        type=int,
                        default=5,
                        help='Maximum number of checkpoints to keep on disk.')
    parser.add_argument('--seed',
                        type=int,
                        default=502,
                        help='Random seed for reproducibility.')
    args = parser.parse_args()
    main(args)












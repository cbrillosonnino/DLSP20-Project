import argparse, os
from pathlib import Path
import shutil

import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import kornia

from data_loading import get_loaders
from loss import Loss, AverageMeter, save_checkpoint
from model_bb import Yo2o

def get_parser():
    parser = argparse.ArgumentParser(description='Flickr30k Training')
    parser.add_argument('-batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('-epoch', type=int, default=30,
                        help='number of training epochs')
    parser.add_argument('-lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('-feature_size', type=int, default=20,
                        help='image grid dimensions')
    parser.add_argument('-num_bboxes', type=int, default=2,
                        help='number of boxes per grid cell')
    parser.add_argument('-lambda_coord', type=float, default=5.0,
                        help='scalar for coordinate loss')
    parser.add_argument('-lambda_noobj', type=float, default=0.5,
                        help='scalar for confidence loss')
    parser.add_argument('-thresh', type=float, default=0.5,
                        help='prediction threshold')
    parser.add_argument('--save', type=str, default=Path.cwd(),
                        help='directory to save logs and models.')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--results', default='results', type=str,
                        help='name of results txt file')

    return parser

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    cudnn.benchmark = True

    args = get_parser().parse_args()

    learning_rate = args.lr
    start_epoch = 0
    max_val_ats = 0

    trainloader, valloader = get_loaders('labeled', batch_size = args.batch_size)

    model = Yo2o(args.feature_size, args.num_bboxes, device).to(device)
    loss_fxn = Loss(args.feature_size, args.num_bboxes, args.lambda_coord, args.lambda_noobj)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            max_val_ats= checkpoint['max_val_ats']
            emodel.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else: print("No checkpoint found at '{}'".format(args.resume))

    if not args.resume:
        file = open(f'{args.save}/{args.results}.txt','a')
        file.write('Train_Loss,Val_ATS \n')
        file.close()

    train_loss = AverageMeter()

    for epoch in range(start_epoch, args.epoch):
        start_epoch += 1
        print('Epoch {}'.format(start_epoch))
        model.train()
        print('training...')
        train_loss.reset()

        for i, (sample, target, road_image) in enumerate(trainloader):
            if i%100 ==  0:
                print('[{}/{}] {}'.format(i,len(trainloader), train_loss.avg))
            batch_size = len(sample)
            img_batch = torch.stack(sample).to(device)
            output = model(img_batch)
            loss = loss_fxn(output, target, device)
            train_loss.update(loss, batch_size)
            model.zero_grad()
            loss.backward()
            optimizer.step()


        model.eval()
        print('validating...')
        ats = 0
        for i, (sample, target, road_image) in enumerate(valloader):
            with torch.no_grad():
                output = model(img_batch)
                ats = loss_fxn.validate(output, target, conf_thresh = args.thresh)
                val_ats += ats

        print('Validation ATS = {}, Learning Rate = {}'.format(ats, learning_rate))
        is_best = max_val_ats < ats
        max_BLEU = max(ats, max_val_ats)


        if epoch+1 % 10 == 0:
            learning_rate /= 10
            for param_group in optimizer.param_groups: param_group['lr'] = learning_rate

        file = open(f'{args.save}/resuts.txt','a')
        file.write('{},{}\n'.format(train_losses.avg,ats))
        file.close()

        save_checkpoint({
            'epoch': epoch + 1, 'model': model.state_dict(),
            'max_val_ats': max_val_ats, 'optimizer' : optimizer.state_dict(),
        }, is_best, args.save)

if __name__ == '__main__': main()

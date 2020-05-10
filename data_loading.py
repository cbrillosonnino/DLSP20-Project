import os
import pickle

import numpy as np
from data_helper import LabeledDataset,UnlabeledDataset
from helper import collate_fn
from torchvision import transforms
import torch


def get_loaders(data_type, image_folder = 'data', 
            annotation_file = 'data/annotation.csv',
                split_folder = 'data_utils', batch_size = 4, extra_info=False):
    """
    Args:
        type (string): 'labeled' or 'unlabeled'
        image_folder (string, optional): the location of the image folders
        annotation_file (string, optional): the location of the annotations
        split_folder (string, optional): the location of the split folder
        batch_size (int, optional): how many samples to load per batch
        extra_info (Boolean, optional): whether you want the extra information
    """

    assert data_type in ['labeled','unlabeled'], "Set correct data_type"

    if data_type == 'labeled':
        train_labeled_scene_index = pickle.load(open(
            os.path.join(split_folder, 'labeled_scene_index_train.p'), 'rb'))
        val_labeled_scene_index = pickle.load(open(
            os.path.join(split_folder, 'labeled_scene_index_val.p'), 'rb'))

        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.698, 0.718, 0.730),
                                     (0.322, 0.313, 0.308))
                ])

        trainset = LabeledDataset(image_folder=image_folder,
                                  annotation_file=annotation_file,
                                  scene_index=train_labeled_scene_index,
                                  transform=transform,
                                  extra_info=extra_info
                                 )
        valset = LabeledDataset(image_folder=image_folder,
                          annotation_file=annotation_file,
                          scene_index=val_labeled_scene_index,
                          transform=transform,
                          extra_info=extra_info
                         )

        trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=2,
                                          collate_fn=collate_fn
                                         )

        valloader = torch.utils.data.DataLoader(valset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=2,
                                  collate_fn=collate_fn
                                 )
    else:
        train_unlabeled_scene_index = pickle.load(open(
            os.path.join(split_folder, 'unlabeled_scene_index_train.p'), 'rb'))
        val_unlabeled_scene_index = pickle.load(open(
            os.path.join(split_folder, 'unlabeled_scene_index_val.p'), 'rb'))

        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.548, 0.597, 0.630),
                                     (0.339, 0.340, 0.342))
                ])

        trainset = UnlabeledDataset(image_folder=image_folder,
                                  first_dim='sample',
                                  scene_index=train_unlabeled_scene_index,
                                  transform=transform,
                                 )
        valset = UnlabeledDataset(image_folder=image_folder,
                          first_dim='sample',
                          scene_index=val_unlabeled_scene_index,
                          transform=transform)

        trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=2
                                         )

        valloader = torch.utils.data.DataLoader(valset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=2
                                 )

    return trainloader, valloader



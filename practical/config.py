import numpy as np

from pathlib import Path
from typing import *

from tqdm.notebook import tqdm

import torch

from monai.transforms import *
from monai.config.type_definitions import KeysCollection

import wandb

# Imports from local files
from transforms import *
from utils import *


# Set random seed
np.random.seed(99)
torch.manual_seed(99)

wandb.login()


class Config():

    def __init__(self):

        # paths
        # directory where img folders are still sorted by domain (but unprocessed OCT images)
        self.name_dir = Path(Path.cwd() / 'data/RETOUCH/TrainingSet-Release/') 
        # already processed OCT images but unsorted by domain (sorting happens in dataset class)
        self.train_dir = Path(Path.cwd() / 'data/Retouch-Preprocessed/train') 
        self.model_path = Path(Path.cwd() / 'models')


        self.source_domains = ['Spectralis', 'Topcon', 'Cirrus']


        # transforms
        self.train_transforms = Compose([
                                        CustomImageLoader(keys=['img', 'label']), # if SVDNA should not be performed, uncomment this and comment the following two lines
                                        #SVDNA(keys=['img'], histogram_matching_degree=.5),
                                        #CustomImageLoader(keys=['label']),
                                        ConvertLabelMaskToChannel(keys=['label'], target_keys=["masks"]),
                                        ExpandChannelDim(keys=['img', 'label']),
                                        ToTensord(keys=['img', 'label', 'masks']),
                                        #Lambdad(keys=['img', 'label', 'masks'], func = lambda x: 2*(x - x.min()) / (x.max() - x.min()) - 1 ),  # -1 to 1 scaling
                                        NormalizeToZeroOne(keys=['img', 'label', 'masks']),
                                        Resized(keys=["img", "label", 'masks'], mode=["area", "nearest-exact", "nearest-exact"], spatial_size=[496, 1024]),
                                        SpatialPadd(keys=['img', 'label', 'masks'], spatial_size=[512, 1024], mode='constant'),
                                        #RandZoomd(keys=["img", "label", 'masks'], mode=["area", "nearest-exact", "nearest-exact"], prob=0.3, min_zoom=0.5, max_zoom=1.5),
                                        #RandAxisFlipd(keys=["img", "label", 'masks'], prob=0.3),
                                        #RandAffined(keys=["img", "label", 'masks'], 
                                        #            prob=0.3, 
                                        #            shear_range=[0, 0],
                                        #            translate_range=[0, 0],
                                        #            rotate_range=[0, 0],
                                        #            mode=["bilinear", "nearest", "nearest"], 
                                        #            padding_mode="zeros"),      
                                        #Debugging(keys=['img', 'label', 'masks']),
                                        ])


        self.val_transforms = Compose([
                                        CustomImageLoader(keys=['img', 'label']),
                                        ConvertLabelMaskToChannel(keys=['label'], target_keys=["masks"]),
                                        ExpandChannelDim(keys=['img', 'label']),
                                        ToTensord(keys=['img', 'label', 'masks']),
                                        NormalizeToZeroOne(keys=['img', 'label', 'masks']),
                                        Resized(keys=["img", "label", 'masks'], mode=["area", "nearest-exact", "nearest-exact"], spatial_size=[496, 1024]),
                                        SpatialPadd(keys=['img', 'label', 'masks'], spatial_size=[512, 1024], mode='constant'),
                                        
                                    ])
        
        
        self.test_transforms = Compose([
                                        CustomImageLoader(keys=['img', 'label']),
                                        ConvertLabelMaskToChannel(keys=['label'], target_keys=["masks"]),
                                        ExpandChannelDim(keys=['img', 'label']),
                                        ToTensord(keys=['img', 'label', 'masks']),
                                        NormalizeToZeroOne(keys=['img', 'label', 'masks']),
                                        Resized(keys=["img", "label", 'masks'], mode=["area", "nearest-exact", "nearest-exact"], spatial_size=[496, 1024]),
                                        SpatialPadd(keys=['img', 'label', 'masks'], spatial_size=[512, 1024], mode='constant'),
                                    ])

        # device
        if torch.backends.mps.is_available():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # models
        self.model_parameters_unet = {
            'spatial_dims': 2,
            'in_channels': 1,
            'out_channels': 3,
            'channels': (16, 32, 64, 128, 256),
            'strides': (2, 2, 2, 2),
            'num_res_units': 2,
            'bias': False,
            'dropout':0.1
        }


        self.encoder_name = "resnet18"


        # optimizer

        self.lr = 1e-4
        self.weight_decay = 0.003


        # hyperparams
        self.batch_size = 3
        self.epochs = 100



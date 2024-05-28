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


class Config():

    def __init__(self, source_domains: List[str] = ['Spectralis', 'Topcon', 'Cirrus']):

        # directory where img folders are still sorted by domain (but unprocessed OCT images)
        self.name_dir = Path(Path.cwd() / 'data/RETOUCH/TrainingSet-Release/') 

        # already processed OCT images but unsorted by domain (sorting happens in dataset class)
        self.train_dir = Path(Path.cwd() / 'data/Retouch-Preprocessed/train') 

        self.default_root_dir = Path(Path.cwd())
        self.model_path = Path(Path.cwd() / 'models')
        self.validation_img_path = Path(Path.cwd() / 'val_predictions')
        self.results_path = Path(Path.cwd() / 'results')

        if not os.path.isdir(self.model_path):
            os.mkdir(Path(Path.cwd() / 'models'))

        if not os.path.isdir(self.validation_img_path):
            os.mkdir(Path(Path.cwd() / 'val_predictions'))

        if not os.path.isdir(self.results_path):
            os.mkdir(Path(Path.cwd() / 'results'))


        self.source_domains = source_domains


        # transforms
        self.train_transforms = Compose([
            #CustomImageLoader(keys=['img', 'label']), # if SVDNA should not be performed, uncomment this and comment the following two lines
            SVDNA(keys=['img'], histogram_matching_degree=.5, source_domains=self.source_domains),
            CustomImageLoader(keys=['label']),
            ConvertLabelMaskToChannel(keys=['label'], target_keys=["masks"]),
            ExpandChannelDim(keys=['img', 'label']),
            ToTensord(keys=['img', 'label', 'masks']),
            #Lambdad(keys=['img', 'label', 'masks'], func = lambda x: 2*(x - x.min()) / (x.max() - x.min()) - 1 ),  # -1 to 1 scaling
            NormalizeToZeroOne(keys=['img', 'label', 'masks']),
            Resized(keys=["img", "label", 'masks'], mode=["area", "nearest-exact", "nearest-exact"], spatial_size=[512, 1024]),
            #RandZoomd(keys=["img", "label", 'masks'], mode=["area", "nearest-exact", "nearest-exact"], prob=0.3, min_zoom=1, max_zoom=1.5),
            #RandAxisFlipd(keys=["img", "label", 'masks'], prob=0.3),
            #RandAffined(keys=["img", "label", 'masks'], 
            #            prob=0.3, 
            #            shear_range=[(-0.2, 0.2), (0.0, 0.0)], 
            #            translate_range=[(-100, 100), (0, 0)],
            #            rotate_range=[-15, 15],
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
            Resized(keys=["img", "label", 'masks'], mode=["area", "nearest-exact", "nearest-exact"], spatial_size=[512, 1024]),
            
        ])
        
        
        self.test_transforms = Compose([
            CustomImageLoader(keys=['img', 'label']),
            ConvertLabelMaskToChannel(keys=['label'], target_keys=["masks"]),
            ExpandChannelDim(keys=['img', 'label']),
            ToTensord(keys=['img', 'label', 'masks']),
            NormalizeToZeroOne(keys=['img', 'label', 'masks']),
            Resized(keys=["img", "label", 'masks'], mode=["area", "nearest-exact", "nearest-exact"], spatial_size=[512, 1024]),
        ])

        # leftover transforms I keep for later
        '''
            #GetMaskPositions(keys=['masks'], target_keys=["mask_positions"]), #We get the layer position, but on the original height
            #LayerPositionToProbabilityMap(["mask_positions"], target_size=(400,400), target_keys=["mask_probability_map"]),

            #Resized(keys=["img", "label", "masks"], mode=["area", "nearest-exact", "nearest-exact"], spatial_size=[400, 400]),
            #Lambdad(keys=['mask_positions'], func = lambda x: x * 400 / 800), #We scale down the positions to have more accurate positions
            #Lambdad(keys=['img'], func = lambda x: np.clip((x - x.mean()) / x.std(), -1, 1)), 
        '''
        

        # device
        if torch.backends.mps.is_available():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'


        # models

        # unet
        self.model_parameters_unet = {
            'spatial_dims': 2,
            'in_channels': 1,
            'out_channels': 4,
            'channels': (16, 32, 64, 128, 256),
            'strides': (2, 2, 2, 2),
            'num_res_units': 2,
            'bias': False,
            'dropout':0.1
        }


        # unet++
        self.model_parameters_unetpp = {
            'encoder_name': 'resnet18',
            'encoder_weights': 'imagenet',
            'decoder_channels': (1024, 512, 256, 128, 64),
            'in_channels': 1,
            'classes': 4,
            'decoder_attention_type': 'scse'
        }


        # optimizer
        self.lr = 1e-4
        self.weight_decay = 0.003


        # lr scheduler
        self.factor = 0.3
        self.patience_lr = 5


        # callbacks
        self.early_stopping_patience = 10


        # hyperparams
        self.batch_size = 3
        self.epochs = 1

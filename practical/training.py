import numpy as np

import os
from typing import *

from tqdm.notebook import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

import monai
from monai.transforms import *
from monai.config.type_definitions import KeysCollection

import segmentation_models_pytorch as smp

import wandb

# Imports from local files
from transforms import *
from dataset import OCTDatasetPrep, MakeDataset
from utils import *
from config import Config

# Set random seed
np.random.seed(99)
torch.manual_seed(99)


wandb.login()


cfg = Config()


def get_model(cfg):

    return smp.UnetPlusPlus(encoder_name=cfg.encoder_name,
                             encoder_weights="imagenet",
                             decoder_channels = (1024, 512, 256, 128, 64),
                             in_channels=1,
                             classes=4,
                             decoder_attention_type="scse")



if not os.path.isdir('models'):
    os.mkdir('models')

if not os.path.isdir('val_predictions'):
    os.mkdir('val_predictions')


print("Device: ", cfg.device)


#model = monai.networks.nets.UNet(**cfg.model_parameters_unet).to(cfg.device)
model = get_model(cfg).to(cfg.device)

criterion = monai.losses.DiceCELoss(sigmoid=True)

optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=5)

train_data, val_data, _ = OCTDatasetPrep(cfg.train_dir).get_datasets(dataset_split=[0.05, 0.006])

train_dataset = MakeDataset(train_data, cfg.train_transforms)
val_dataset = MakeDataset(val_data, cfg.val_transforms)
#test_dataset = MakeDataset(test_data, cfg.test_transforms)

train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4)
#test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)



train_the_thing = False

if train_the_thing:

    run_description = 'UNet++ with Resnet18 encoder, 5% of training data, 0.05 learning rate, 0.0001 weight decay, 0.3 lr scheduler factor, 5 patience'


    #model = load_model(model, cfg.model_path / 'model_trial_withbg.pth')


    wandb_config = {
        'batch_size': cfg.batch_size,
        'lr': optimizer.param_groups[0]['lr'],
        'epochs': cfg.epochs,
        'device': cfg.device,
        'model': 'UNet++ Resnet18 encoder',
        'dataset': 'Retouch',
        'model_parameters': cfg.model_parameters_unet,
    }


    with wandb.init(project='PracticalWorkinAI', 
                    config=wandb_config,
                    name=run_description) as run:
        
        wandb_config = wandb.config
        
        model = train(model, 
                    train_loader, val_loader, 
                    criterion, optimizer, scheduler, cfg.device, 
                    epochs=cfg.epochs, 
                    save_path= cfg.model_path / 'model_trial_withbg.pth')
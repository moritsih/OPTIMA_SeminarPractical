
import torch
import torch.optim as optim
import torch
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp

import lightning as L
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from config import Config
from dataset import *
from utils import compute_dice_score, make_results_table, DiceCELossSplitter, plot_img_label_pred
from transforms import ImageVisualizer

import monai
from monai.transforms import *

from tabulate import tabulate
import matplotlib.pyplot as plt

seed_everything(99, workers=True)


train_data, val_data, test_data = OCTDatasetPrep(cfg.train_dir,
                                         source_domains = [
                                             'Spectralis',
                                             'Topcon',      # define all three domains as "source" -> supervised setting
                                             'Cirrus'
                                         ]).get_datasets(dataset_split=[0.001, 0.0002, 0.1])


train_dataset = MakeDataset(train_data, cfg.train_transforms)
val_dataset = MakeDataset(val_data, cfg.val_transforms)
test_dataset = MakeDataset(test_data, cfg.test_transforms)

train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=7, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=7, persistent_workers=True)
test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=7, persistent_workers=True)


class LitUNetPlusPlus(L.LightningModule):
    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        #self.save_hyperparameters(ignore=['model'])
        self.save_hyperparameters()
        self.model = model
        self.loss_func = DiceCELossSplitter(include_background=True, sigmoid=True, lambda_ce=0.5)
    
    def training_step(self, batch, batch_idx):
        inputs = batch['img'].to(self.cfg.device)
        #labels = sample['label'].to(device)
        masks = batch['masks'].to(self.cfg.device)

        outputs = self.model(inputs)
        dice_loss, ce_loss, total_loss = self.loss_func(outputs, masks)

        print(f"Total loss: {total_loss} | Dice loss: {dice_loss} | CE loss: {ce_loss}")

        self.log('train_loss_dice', dice_loss)
        self.log('train_loss_ce', ce_loss)
        self.log('train_loss_total', total_loss)
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        inputs = batch['img'].to(self.cfg.device)
        #labels = sample['label'].to(device)
        masks = batch['masks'].to(self.cfg.device)

        outputs = self.model(inputs)
        # try splitting losses
        _, _, total_loss = self.loss_func(outputs, masks)

        output_to_save = torch.sigmoid(outputs[:5])

        # thresholding
        output_to_save[output_to_save > 0.5] = 1
        output_to_save[output_to_save <= 0.5] = 0

        for i in range(output_to_save.shape[0]):
            img_path = os.path.join(cfg.validation_img_path, f"b{batch_idx}img{i+1}.png")
            cv2.imwrite(img_path, output_to_save[i, :, :, :].permute(1, 2, 0).cpu().numpy() * 255)

        self.log('val_loss_total', total_loss)
        return total_loss
    
    def test_step(self, batch, batch_idx):
        inputs = batch['img'].to(self.cfg.device)
        #labels = sample['label'].to(device)
        masks = batch['masks'].to(self.cfg.device)

        outputs = self.model(inputs)
        #loss = self.loss_func(outputs, masks)
        # try splitting losses
        _, _, total_loss = self.loss_func(outputs, masks)
        self.log('test_loss_total', total_loss)
        return total_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.cfg.factor, patience=self.cfg.patience_lr)
        return {"optimizer": optimizer, 
                "lr_scheduler": {'scheduler': scheduler, 'monitor': 'val_loss_total'}}
    
    def forward(self, x):
        return self.model(x)
    
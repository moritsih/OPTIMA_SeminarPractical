
import torch
import torch.optim as optim
import torch

import lightning as L
from lightning.pytorch import seed_everything
import torchmetrics as tm
from lightning.pytorch.callbacks import Callback

import monai
import os
import cv2
from pathlib import Path
import pandas as pd
from tabulate import tabulate

from lightning.pytorch.plugins.environments import SLURMEnvironment
SLURMEnvironment.detect = lambda: False


class LitUNetPlusPlus(L.LightningModule):
    def __init__(self, cfg, model, experiment_name):
        super().__init__()
        self.cfg = cfg

        self.save_hyperparameters()
        self.model = model
        self.experiment_name = experiment_name

        self.loss_func1 = monai.losses.GeneralizedDiceLoss(include_background=False, sigmoid=True)
        self.loss_func2 = torch.nn.BCEWithLogitsLoss()

        # self.loss_func = monai.losses.GeneralizedDiceFocalLoss(include_background=False, sigmoid=True)

        # several metrics
        avg = 'weighted' # weighted: calculates statistics for each label and computes weighted average using their support
        num_classes = 2
        task = 'binary'

        # validation
        self.val_accuracy = tm.classification.Accuracy(task=task, num_classes=num_classes)
        self.val_f1 = tm.classification.F1Score(task=task, num_classes=num_classes)
        self.val_precision = tm.classification.Precision(task=task, average=avg, num_classes=num_classes)
        self.val_recall = tm.classification.Recall(task=task, average=avg, num_classes=num_classes)
        self.val_specificity = tm.classification.Specificity(task=task, average=avg, num_classes=num_classes)

        # test
        self.test_accuracy = tm.classification.Accuracy(task=task, num_classes=num_classes)
        self.test_f1 = tm.classification.F1Score(task=task, num_classes=num_classes)
        self.test_precision = tm.classification.Precision(task=task, average=avg, num_classes=num_classes)
        self.test_recall = tm.classification.Recall(task=task, average=avg, num_classes=num_classes)
        self.test_specificity = tm.classification.Specificity(task=task, average=avg, num_classes=num_classes)


        self.results = {"Model": [], 
                        "Task": [], 
                        "Accuracy": [], 
                        "F1": [], 
                        "Precision": [], 
                        "Recall": [], 
                        "Specificity": []}
        
        self.tasks = ["Background", "IRF", "SRD", "PED"]

    def training_step(self, batch, batch_idx):
        inputs = batch['img']
        masks = batch['masks']

        outputs = self.model(inputs)

        dice_loss = self.loss_func1(outputs, masks)
        ce_loss = self.loss_func2(outputs, masks)
        total_loss = dice_loss + 0.5 * ce_loss

        self.log('train_loss_dice', dice_loss.item())
        self.log('train_loss_ce', ce_loss.item())
        self.log('train_loss_total', total_loss.item())

        return total_loss
    
    def validation_step(self, batch, batch_idx):
        inputs = batch['img']
        masks = batch['masks']

        outputs = self.model(inputs)

        dice = self.loss_func1(outputs, masks)
        ce = self.loss_func2(outputs, masks)
        total_loss = dice + 0.5 * ce

        outputs_nobg = outputs[:, 1:, :, :]
        masks_nobg = masks[:, 1:, :, :]

        #output_to_save = torch.sigmoid(outputs_nobg[:5])

        # thresholding
        #output_to_save[output_to_save > 0.5] = 1
        #output_to_save[output_to_save <= 0.5] = 0

        #for i in range(output_to_save.shape[0]):
        #    img_path = os.path.join(self.cfg.validation_img_path, f"epoch{self.current_epoch}")
        #    if not os.path.isdir(img_path): os.mkdir(img_path)
        #    img = os.path.join(img_path, f"b{batch_idx}img{i+1}.png")
        #    cv2.imwrite(img, output_to_save[i].permute(1, 2, 0).cpu().numpy() * 255)

        self.log('val_loss_total', total_loss)
        self.log('val_loss_dice', dice)
        self.log('val_loss_ce', ce)

        self.log('val_accuracy', self.val_accuracy(outputs_nobg, masks_nobg.long()))
        self.log('val_f1', self.val_f1(outputs_nobg, masks_nobg.long()))
        self.log('val_precision', self.val_precision(outputs_nobg, masks_nobg.long()))
        self.log('val_recall', self.val_recall(outputs_nobg, masks_nobg.long()))
        self.log('val_specificity', self.val_specificity(outputs_nobg, masks_nobg.long()))

        return total_loss
    
    def test_step(self, batch, batch_idx):
        inputs = batch['img']
        masks = batch['masks']

        outputs = self.model(inputs)

        dice = self.loss_func1(outputs, masks)
        ce = self.loss_func2(outputs, masks)
        total_loss = dice + 0.5 * ce

        self.log('test_loss_dice', dice)
        self.log('test_loss_ce', ce)
        self.log('test_loss_total', total_loss)

        torch.sigmoid_(outputs)

        # calculate metrics for each channel and record separately in pandas df
        for channel in range(outputs.shape[1]):

            outputs_channel = outputs[:, channel, :, :]
            masks_channel = masks[:, channel, :, :]

            accuracy = self.test_accuracy(outputs_channel, masks_channel)
            f1 = self.test_f1(outputs_channel, masks_channel)
            precision = self.test_precision(outputs_channel, masks_channel)
            recall = self.test_recall(outputs_channel, masks_channel)
            specificity = self.test_specificity(outputs_channel, masks_channel)

            self.results["Model"].append(self.experiment_name)
            self.results["Task"].append(self.tasks[channel])
            self.results["Accuracy"].append(accuracy.item())
            self.results["F1"].append(f1.item())
            self.results["Precision"].append(precision.item())
            self.results["Recall"].append(recall.item())
            self.results["Specificity"].append(specificity.item())
        
        return total_loss

    def configure_optimizers(self):
        #optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.cfg.factor, patience=self.cfg.patience_lr)
        return {"optimizer": optimizer, 
                "lr_scheduler": {'scheduler': scheduler, 'monitor': 'val_loss_total'}}
    
    def forward(self, x):
        return self.model(x)
    



class SaveInitialModelCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        trainer.save_checkpoint(os.path.join(trainer.default_root_dir, "models", pl_module.experiment_name, "initial_model.ckpt"))



class AggregateTestingResultsCallback(Callback):
    def on_test_end(self, trainer, pl_module):


        if not os.path.exists(f"{trainer.default_root_dir}/results/{pl_module.experiment_name}"):
            os.makedirs(f"{trainer.default_root_dir}/results/{pl_module.experiment_name}")
        
        save_path = f"{trainer.default_root_dir}/results/{pl_module.experiment_name}"



        self.results = pd.DataFrame.from_dict(pl_module.results)
        
        # group by condition and calculate mean
        grouped_means = self.results.groupby(["Task"]).agg({
            "Model": "first",
            "Dice": "mean",
            "Accuracy": "mean",
            "F1": "mean",
            "Precision": "mean",
            "Recall": "mean",
            "Specificity": "mean"
        })

        # print the results
        print(tabulate(grouped_means, headers="keys", tablefmt="pretty"))


        self.results.to_csv(f"{save_path}/results_{pl_module.experiment_name}.csv")
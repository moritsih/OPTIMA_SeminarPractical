import numpy as np
import os
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

from monai.transforms import *
from transforms import *

import monai
import pandas as pd
import tabulate

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning.pytorch.callbacks import Callback
from tqdm.notebook import tqdm
import wandb

np.random.seed(99)
torch.manual_seed(99)


class AggregateTestingResultsCallback(Callback):
    def on_test_end(self, trainer, pl_module):

        self.results = pd.DataFrame.from_dict(pl_module.results)
        
        # group by condition and calculate mean and std in separate columns
        grouped_results = self.results.groupby(["Model", "Condition"]).agg({
            "Accuracy": ["mean", "std"],
            "F1": ["mean", "std"],
            "Precision": ["mean", "std"],
            "Recall": ["mean", "std"],
            "Specificity": ["mean", "std"],
            "Model": "first"
        })
        
        # print the results
        print(tabulate(grouped_results, headers="keys", tablefmt="pretty"))

        try:
            self.results.to_csv(f"{pl_module.cfg.results_path}results_{pl_module.experiment_name}.csv")
        except:
            self.results.to_csv(f"results_{pl_module.experiment_name}.csv")


'''
Rewrite the loss function used so I can log the two losses separately
'''

class DiceCELossSplitter(monai.losses.DiceCELoss):

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 nor the same as input.

        """
        if len(input.shape) != len(target.shape):
            raise ValueError(
                "the number of dimensions for input and target should be the same, "
                f"got shape {input.shape} and {target.shape}."
            )

        dice_loss = self.dice(input, target)
        ce_loss = self.ce(input, target) if input.shape[1] != 1 else self.bce(input, target)
        total_loss: torch.Tensor = self.lambda_dice * dice_loss + self.lambda_ce * ce_loss

        return dice_loss, ce_loss, total_loss
    

# other utility functions

def map_grayscale_to_channels(image):
    # Create an empty array with shape (height, width, 4)
    
    mapped_image = np.zeros((image.shape[0], image.shape[1], len(np.unique(image))), dtype=np.uint8)

    for i, pixel_value in enumerate(np.unique(image)):
        print("Pixel value: ", pixel_value)
        mapped_image[image == pixel_value, i] = 255


    if len(np.unique(image)) == 1: # in case an empty (black) image label is encountered, the unique values will be 1
        print("Empty image encountered")
        print(mapped_image.shape)
        return mapped_image
    
    return mapped_image[:, :, 1:4]




def map_grayscale_to_channels_four_channel(image):
    # Create an empty array with shape (height, width, 4)
    
    mapped_image = np.zeros((image.shape[0], image.shape[1], len(np.unique(image))), dtype=np.uint8)

    for i, pixel_value in enumerate(np.unique(image)):
        mapped_image[image == pixel_value, i] = 255


    if len(np.unique(image)) == 1: # in case an empty (black) image label is encountered, the unique values will be 1
        return mapped_image
    
    return mapped_image[:, :, 1:4]


def plot_img_label_pred(img, pred, mask):
    
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    
    img = img.squeeze().cpu().detach().numpy()
    mask = mask.squeeze().permute(1,2,0).cpu().detach().numpy()[:, :, 1:]
    pred = pred.squeeze().permute(1,2,0).cpu().detach().numpy()[:, :, 1:]

    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('Image')
    ax[1].imshow(pred, cmap='gray')
    ax[1].set_title('Prediction')
    ax[2].imshow(mask, cmap='gray')
    ax[2].set_title('Ground Truth')
    plt.show()
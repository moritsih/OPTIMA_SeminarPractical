import numpy as np
import os
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

from monai.transforms import *
from transforms import *

import monai

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
#from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
import wandb

np.random.seed(99)
torch.manual_seed(99)

# utils for result output

def compute_dice_score(pred, mask):

    '''
    Compute the dice score for each class in the mask
    '''

    dice_scores = []

    for i in range(1, mask.shape[1]):
        dice_score = 1 - monai.losses.DiceLoss(sigmoid=True)(pred[:, i], mask[:, i])
        dice_scores.append(dice_score.item())

    return dice_scores



def make_results_table(model, dataset, device, cfg, model_name):
    '''
    Using tabulate, the model is applied to the test dataset and the mean dice score is calculated for each class.
    The results are then saved in a table and returned.
    '''
    model.eval()

    results = []

    with torch.no_grad():  # Ensure no gradients are calculated
        for i, sample in enumerate(tqdm(dataset)):
            img = sample['img'].unsqueeze(0).to(device)
            mask = sample['masks'].unsqueeze(0).to(device)

            pred = model(img)
            pred = torch.sigmoid(pred)

            # thresholding
            pred[pred > 0.5] = 1
            pred[pred <= 0.5] = 0

            dice_scores = compute_dice_score(pred, mask)
            results.append(dice_scores) 

            del img, mask, pred  # Explicitly delete tensors to free memory
            torch.cuda.empty_cache()  # Free up unused memory

    results = np.array(results)
    mean_dice_scores = np.round(np.mean(results, axis=0), 3)

    table = [model_name, mean_dice_scores[0], mean_dice_scores[1], mean_dice_scores[2]]

    return table



table = [['Model', 'IRF', 'SRF', 'PED']]





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
import numpy as np
import os
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

from monai.transforms import *
from transforms import *

import monai
import pandas as pd
from tabulate import tabulate

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning.pytorch.callbacks import Callback
from tqdm.notebook import tqdm
import wandb

np.random.seed(99)
torch.manual_seed(99)



class SaveInitialModelCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        trainer.save_checkpoint(os.path.join(trainer.default_root_dir, "models", "initial_model.ckpt"))



class AggregateTestingResultsCallback(Callback):
    def on_test_end(self, trainer, pl_module):

        self.results = pd.DataFrame.from_dict(pl_module.results)
        
        # group by condition and calculate mean
        grouped_means = self.results.groupby(["Task"]).agg({
            "Model": "first",
            "Accuracy": "mean",
            "F1": "mean",
            "Precision": "mean",
            "Recall": "mean",
            "Specificity": "mean"
        })

        # group by condition and calculate std
        grouped_std = self.results.groupby(["Task"]).agg({
            "Model": "first",
            "Accuracy": "std",
            "F1": "std",
            "Precision": "std",
            "Recall": "std",
            "Specificity": "std"
        })

        # print the results
        print(tabulate(grouped_means, headers="keys", tablefmt="pretty"))

        if not os.path.exists(f"{Path.cwd()}/results/{pl_module.experiment_name}"):
            os.makedirs(f"{Path.cwd()}/results/{pl_module.experiment_name}")
        
        save_path = f"{Path.cwd()}/results/{pl_module.experiment_name}"

        self.results.to_csv(f"{save_path}/results_raw_{pl_module.experiment_name}.csv")
        grouped_means.to_csv(f"{save_path}/results_mean_{pl_module.experiment_name}.csv")
        grouped_std.to_csv(f"{save_path}/results_std_{pl_module.experiment_name}.csv")

    

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
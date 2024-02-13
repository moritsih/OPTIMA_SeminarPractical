import numpy as np
import monai
import cv2
import torch
from monai.transforms import *
from pathlib import Path
import os
import glob


'''
1. Load RETOUCH dataset
    - folder
https://docs.monai.io/en/stable/networks.html#basicunetplusplus


First of all, I need to get the data and create a pipeline for using transformations during training.
I need to create a pipeline for the training data and another for the validation data.
I need to get a UNet++ from monai and train it with the training data.
I need to get SVDNA and apply it to the data as the first transformation
Finally, during training, first SVDNA is applied, then any other transformation.



'''

data_folder = Path('data')


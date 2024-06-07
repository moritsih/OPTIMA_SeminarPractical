import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch

np.random.seed(99)
torch.manual_seed(99)



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


def str2bool(v):
    # helper function to convert string to boolean for argument parsing
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
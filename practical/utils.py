import numpy as np
import os

import torch
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb

np.random.seed(99)
torch.manual_seed(99)


# important for training

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch_idx, sample in enumerate(train_loader):
        inputs = sample['img'].to(device)
        #labels = sample['label'].to(device)
        masks = sample['masks'].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        #print(f"Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(train_loader)
    return epoch_loss



def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            inputs = sample['img'].to(device)
            #labels = sample['label'].to(device)
            masks = sample['masks'].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, masks)

            running_loss += loss.item()

    epoch_loss = running_loss / len(val_loader)
    wandb.log({"val_loss": epoch_loss})
    #print(f"Validation loss: {epoch_loss:.4f}")
    return epoch_loss


def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            inputs = sample['img'].to(device)
            #labels = sample['label'].to(device)
            masks = sample['masks'].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, masks)

            running_loss += loss.item()

    epoch_loss = running_loss / len(test_loader)
    wandb.log({"test_loss": epoch_loss})
    #print(f"Test loss: {epoch_loss:.4f}")
    return epoch_loss



def save_model(model, loss, best_loss, save_path):
    if loss < best_loss:
        best_loss = loss
        torch.save(model.state_dict(), save_path)
        #print("Model saved.")
    return best_loss



def load_model(model, load_path):
    model.load_state_dict(torch.load(load_path))
    return model



def train(model, train_loader, val_loader, criterion, optimizer, device, epochs, save_path):

    wandb.watch(model, criterion, log="all", log_freq=10)
    best_loss = float('inf')
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)

    for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
        #print(f"Epoch {epoch+1}/{epochs}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        best_loss = save_model(model, val_loss, best_loss, save_path)

        wandb.log({"train_loss": train_loss, 
                   "val_loss": val_loss,
                   "lr": optimizer.param_groups[0]['lr'],
                   "epoch": epoch+1,
                   "best_loss": best_loss})

    return model






# other utility functions

def map_grayscale_to_channels(image):
    # Create an empty array with shape (height, width, 4)
    
    mapped_image = np.zeros((image.shape[0], image.shape[1], len(np.unique(image))), dtype=np.uint8)

    for i, pixel_value in enumerate(np.unique(image)):
        mapped_image[image == pixel_value, i] = 255


    if len(np.unique(image)) == 1: # in case an empty (black) image label is encountered, the unique values will be 1
        return mapped_image
    
    return mapped_image[:, :, 1:4]
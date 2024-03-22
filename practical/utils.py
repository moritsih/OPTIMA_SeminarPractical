import numpy as np
import os

import torch
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

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

        print(f"Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

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
    print(f"Validation loss: {epoch_loss:.4f}")
    return epoch_loss



def save_model(model, loss, best_loss, save_path):
    if loss < best_loss:
        best_loss = loss
        torch.save(model.state_dict(), save_path)
        print("Model saved.")
    return best_loss



def load_model(model, load_path):
    model.load_state_dict(torch.load(load_path))
    return model



def train(model, train_loader, val_loader, criterion, optimizer, device, epochs, save_path):
    writer = SummaryWriter()
    best_loss = float('inf')
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)

    for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        best_loss = save_model(model, val_loss, best_loss, save_path)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
    writer.close()
    return model






# other utility functions

def map_grayscale_to_channels(image):
    # Create an empty array with shape (height, width, 4)
    mapped_image = np.zeros((image.shape[0], image.shape[1], len(np.unique(image))), dtype=np.uint8)

    for i, pixel_value in enumerate(np.unique(image)):
        mapped_image[image == pixel_value, i] = 255

    return mapped_image[:, :, 1:4]
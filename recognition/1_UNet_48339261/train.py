"""
train.py
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split

from modules import DiceLoss, SimpleUNet 
from dataset import HipMRIDataset
from utils import show_epoch_predictions, plot_loss, calculate_dice_score

import numpy as np
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, test_dataset, epochs=3, lr=0.001, visualize_every=1):
    model.to(device)
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    print(f"Train Starting for {epochs} epochs")

    for epoch in range(epochs):
        model.train() 
        epoch_loss = 0

        for batch_idx, (images, masks) in enumerate(train_loader):

            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            predictions_squeezed = outputs[:, 0]  # (C, H, W)
            loss = criterion(predictions_squeezed, masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} Complete: Avg Loss = {avg_loss:.4f}")

        # Visualize predictions after each epoch (or every few epochs)
        if (epoch) % visualize_every == 0:
            show_epoch_predictions(model, test_dataset, epoch + 1, n=3)

    print("Training complete with enhanced U-Net")
    plot_loss(losses)
    return losses


if __name__ == "__main__":
    
    print(f"Starting training (train.py) ---")
    
    DATA_DIR = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data" 
    MODEL_SAVE_PATH = "hipmri_unet_model.pth" 
    EPOCHS = 20           
    LEARNING_RATE = 0.001     
    BATCH_SIZE = 16           
    RESIZE_TO = (128, 128)  
    PROSTATE_LABEL = 5      
    VALIDATION_SPLIT = 0.2  

    dataset = HipMRIDataset(
        data_dir=DATA_DIR,
        resize_to=RESIZE_TO,
        prostate_label_value=PROSTATE_LABEL
    )
    
    val_size = int(len(dataset) * VALIDATION_SPLIT)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], 
                                               generator=torch.Generator().manual_seed(42))
    
    print(f"Dataset split: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4 
    )

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = SimpleUNet(in_channels=1, out_channels=1).to(device)

    training_losses = train(
        model=model,
        train_loader=train_loader,
        test_dataset=val_dataset, 
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        visualize_every=5
    )

    print("Traning Complete")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved in: {MODEL_SAVE_PATH}")

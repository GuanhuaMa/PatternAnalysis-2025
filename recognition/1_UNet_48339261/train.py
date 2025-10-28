"""
train.py
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

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
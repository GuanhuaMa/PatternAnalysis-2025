"""
train.py
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from modules import DiceLoss, SimpleUNet 
from dataset import HipMRIDataset
from utils import show_epoch_predictions, plot_loss, calculate_dice_score
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, val_dataset, epochs=3, lr=0.001, visualize_every=1):
    """
    param val_dataset: validation set
    """
    model.to(device)
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    print(f"Train Starting for {epochs} epochs")

    for epoch in range(epochs):
        model.train() 
        epoch_loss = 0
        
        # show process bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

        for batch_idx, (images, masks) in enumerate(progress_bar):

            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            predictions_squeezed = outputs[:, 0]
            loss = criterion(predictions_squeezed, masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
            # show batch loss in prcocess bar
            progress_bar.set_postfix(batch_loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} Complete: Avg Loss = {avg_loss:.4f}")

        # visualization
        if (epoch + 1) % visualize_every == 0 or (epoch + 1) == epochs:
            show_epoch_predictions(model, val_dataset, epoch + 1, n=3)

    print("Training complete with enhanced U-Net")
    plot_loss(losses)
    return losses

if __name__ == "__main__":
    
    print(f"Starting training (train.py) ---")
    
    DATA_DIR = "/content/drive/MyDrive/Colab-Notebooks/UNet_task3_48339261/keras_slices_data" 
    MODEL_SAVE_PATH = "/content/drive/MyDrive/Colab-Notebooks/UNet_task3_48339261/hipmri_unet_model.pth"

    EPOCHS = 20           
    LEARNING_RATE = 0.001     
    BATCH_SIZE = 16           
    RESIZE_TO = (128, 128)  
    PROSTATE_LABEL = 5     

    # load and split dataset
    print(f"Loading dataset from {DATA_DIR} ...")
    
    # load the split train dataset
    train_dataset = HipMRIDataset(
        data_dir=DATA_DIR,
        subset="train",
        resize_to=RESIZE_TO,
        prostate_label_value=PROSTATE_LABEL
    )
    
    # load the split validate dataset
    val_dataset = HipMRIDataset(
        data_dir=DATA_DIR,
        subset="validate",
        resize_to=RESIZE_TO,
        prostate_label_value=PROSTATE_LABEL
    )
    
    print(f"Dataset loaded {len(train_dataset)} train dataset, {len(val_dataset)} validate sataset")

    # DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=2 
    )

    model = SimpleUNet(in_channels=1, out_channels=1).to(device)

    # start training
    training_losses = train(
        model=model,
        train_loader=train_loader,
        val_dataset=val_dataset,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        visualize_every=5 
    )

    print("Training Complete")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to: {MODEL_SAVE_PATH}")
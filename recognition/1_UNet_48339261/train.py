import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

def train(model, train_loader, test_dataset, epochs=3, lr=0.001, visualize_every=1):

    losses = []

    criterion = DiceLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Train Starting for {epochs} epochs")

    for epoch in range(epochs):
        model.train() 
        epoch_loss = 0

        for batch_idx, (images, masks) in enumerate(train_loader):

            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            pred_pet = outputs[:, 0]  # (C, H, W)
            loss = criterion(pred_pet, masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} Complete: Avg Loss = {avg_loss:.4f}")

    print("Training complete with enhanced U-Net")
    return losses
import torch
import numpy as np
import matplotlib.pyplot as plt

def calculate_dice_score(pred_binary, true_mask):
    """calculate Dice similarity coefficient"""

    if isinstance(pred_binary, torch.Tensor):
        pred_binary = pred_binary.cpu().numpy()
    if isinstance(true_mask, torch.Tensor):
        true_mask = true_mask.cpu().numpy()

    pred_binary = pred_binary.flatten()
    true_mask = true_mask.flatten()

    # + 1e-6 prevent the denominator is 0
    intersection = (pred_binary * true_mask).sum()
    dice_score = (2. * intersection + 1e-6) / (pred_binary.sum() + true_mask.sum() + 1e-6)

    return dice_score


def show_epoch_predictions(model, dataset, epoch, n=3):
    """MRI"""
    model.eval()

    fig, axes = plt.subplots(3, n, figsize=(12, 9))
    fig.suptitle(f'Train the prediction results after the {epoch} round', fontsize=16)

    with torch.no_grad():
        indices = np.random.choice(len(dataset), n, replace=False)

        for i, idx in enumerate(indices):
            image, true_mask = dataset[idx] 
            pred = model(image.unsqueeze(0).to(device))

            pred_prob = pred[0, 0].cpu().numpy()
            pred_binary = (pred_prob > 0.5).astype(int)

            # Show original image 
            img_display = image.squeeze().cpu().numpy()
            axes[0, i].imshow(img_display, cmap='gray')
            axes[0, i].set_title(f'Original {idx})')
            axes[0, i].axis('off')

            # Show ground truth binary mask
            axes[1, i].imshow(true_mask.cpu().numpy(), cmap='gray')
            axes[1, i].set_title(f'Ground Truth (前列腺)')
            axes[1, i].axis('off')

            # Show prediction
            dice = calculate_dice_score(pred_binary, true_mask)
            axes[2, i].imshow(pred_binary, cmap='gray')
            axes[2, i].set_title(f'Prediction') 
            axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()

    model.train()


def plot_loss(losses):
    """Plot train loss curve"""
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.title('Training Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Dice Loss)')
    plt.legend()
    plt.grid(True)
    plt.show() 

    
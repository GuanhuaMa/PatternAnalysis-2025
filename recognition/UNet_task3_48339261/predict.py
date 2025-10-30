import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils import calculate_dice_score
from modules import SimpleUNet
from dataset import HipMRIDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Predict.py: Using device: {device}")

def show_predictions(model, dataset, title="Final segmentation results (HipMRI)", n=3):
    model.eval() 
    
    fig, axes = plt.subplots(3, n, figsize=(12, 9))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    with torch.no_grad():
        if len(dataset) < n:
            n = len(dataset)
            print(f"The siza of Dataset ({len(dataset)}) smaller than n ({n}), show {n} samples.")
            
        indices = np.random.choice(len(dataset), n, replace=False)
        
        for i, idx in enumerate(indices):
            # (1,H,W) & (H,W)
            image, true_mask = dataset[idx]
            # (1,H,W) -> (1,1,H,W)
            pred = model(image.unsqueeze(0).to(device))
            # (1,1,H,W) -> (H,W)
            pred_prob = pred[0, 0].cpu().numpy()
            pred_binary = (pred_prob > 0.5).astype(int)

            # Orifinal (gray)
            img_display = image.squeeze().cpu().numpy()
            axes[0, i].imshow(img_display, cmap='gray')
            axes[0, i].set_title(f'Original {idx})', fontweight='bold')
            axes[0, i].axis('off')

            # Ground Truth
            axes[1, i].imshow(true_mask.cpu().numpy(), cmap='gray')
            axes[1, i].set_title(f'Ground Truth (Prostate)', fontweight='bold')
            axes[1, i].axis('off')

            # Prediction
            dice = calculate_dice_score(pred_binary, true_mask) 
            axes[2, i].imshow(pred_binary, cmap='gray')
            axes[2, i].set_title(f'Pridiction {dice:.3f})', fontweight='bold')
            axes[2, i].axis('off')

    plt.tight_layout()
    save_path = "final_predictions.png"
    plt.savefig(save_path)
    print(f"Prediction Result saved in: {save_path}")
    plt.close(fig)

if __name__ == '__main__':
    print("Runing predict.py...")

    DATA_DIR = "/content/drive/MyDrive/Colab-Notebooks/UNet_task3_48339261/keras_slices_data" 
    MODEL_SAVE_PATH = "/content/drive/MyDrive/Colab-Notebooks/UNet_task3_48339261/hipmri_unet_model.pth"
    RESIZE_TO = (128, 128) 
    PROSTATE_LABEL = 5 
    NUM_EXAMPLES_TO_SHOW = 3

    # Chenk the file exists
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Error: can't find file '{MODEL_SAVE_PATH}'。")
    else:
        print(f"Loading the model: {MODEL_SAVE_PATH}")
        model = SimpleUNet(in_channels=1, out_channels=1).to(device)

        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        
        print(f"Loading the test set: {DATA_DIR} (subset=test)")
        test_dataset = HipMRIDataset(
            data_dir=DATA_DIR,
            subset="test",
            resize_to=RESIZE_TO,
            prostate_label_value=PROSTATE_LABEL
        )
        
        if len(test_dataset) > 0:
            print(f"Generating {NUM_EXAMPLES_TO_SHOW} prediction examples...")
            show_predictions(model, test_dataset, n=NUM_EXAMPLES_TO_SHOW)
        else:
            print("Error: Dataset is empty.")
import torch
import os

from modules import SimpleUNet
from dataset import HipMRIDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Predict.py: Using device: {device}")



if __name__ == '__main__':
    print("Runing predict.py...")

    DATA_DIR = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
    MODEL_SAVE_PATH = "hipmri_unet_model.pth" 
    RESIZE_TO = (128, 128) 
    PROSTATE_LABEL = 5 
    NUM_EXAMPLES_TO_SHOW = 3

    # visuliasation
    pass
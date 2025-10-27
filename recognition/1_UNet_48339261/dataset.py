import torch
from torch.utils.data import Dataset
import os
import glob

class HipMRIDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir

        self.image_files = []
        self.mask_files = []

        image_paths = sorted(glob.glob(os.path.join(data_dir, "*_image.nii.gz")))

        for img_path in image_paths:
            mask_path = img_path.replace("_image.nii.gz", "_mask.nii.gz")
            if os.path.exists(mask_path):
                self.image_files.append(img_path)
                self.mask_files.append(mask_path)

        if len(self.image_files) == 0:
            print(f"Warning: No matching files found in {data_dir}")
        else:
            print(f"Found {len(self.image_files)} image/mask pairs.")

    def __len__(self):

        return len(self.image_files) 
    

    def __getitem__(self, idx):

        return None

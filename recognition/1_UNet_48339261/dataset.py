import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import os
import glob

class HipMRIDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.prostate_label_value = prostate_label_value

        self.image_files = []
        self.mask_files = []

        image_paths = sorted(glob.glob(os.path.join(data_dir, "*_image.nii.gz")))

        # match the Mask file according to the image name
        for img_path in image_paths:
            mask_path = img_path.replace("_image.nii.gz", "_mask.nii.gz")
            if os.path.exists(mask_path):
                self.image_files.append(img_path)
                self.mask_files.append(mask_path)

        if len(self.image_files) == 0:
            print(f"No matching files found in {data_dir}")
        else:
            print(f"Found {len(self.image_files)} image/mask pairs.")

    def __len__(self):

        return len(self.image_files) 
    

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        mask_path = self.mask_files[idx]

        image_nii = nib.load(image_path)
        mask_nii = nib.load(mask_path)

        image = image_nii.get_fdata().astype(np.float32)
        mask = mask_nii.get_fdata().astype(np.uint8)

        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(mask)

        # add chanel dimention
        if image_tensor.ndim == 2:
            image_tensor = image_tensor.unsqueeze(0) # 变为 [1, H, W]
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)   # 变为 [1, H, W]

        # Z-score normalization
        mean = image_tensor.mean()
        std = image_tensor.std()
        if std > 1e-6:
            image_tensor = (image_tensor - mean) / std
        else:
            image_tensor = image_tensor - mean

        binary_mask = (mask_tensor == self.prostate_label_value).long()
        
        return image_tensor, binary_mask

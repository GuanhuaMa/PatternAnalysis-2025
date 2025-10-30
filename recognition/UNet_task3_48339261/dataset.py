"""
dataset.py
"""
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import torch
import os
import glob
import torchvision.transforms.functional as TF

class HipMRIDataset(Dataset):
    """
    keras_slices_train/        : traning figure
    keras_slices_seg_train/    : traning mask
    keras_slices_validate/     : validate figure
    keras_slices_seg_validate/ : validate mask
    keras_slices_test/         : test figure
    keras_slices_seg_test/     : test mask
    """
    def __init__(self, data_dir, subset="train", prostate_label_value=5, resize_to=None):
        self.subset = subset # 'train', 'validate' or 'test'
        self.data_dir = data_dir # the root path of keras_slices_data
        self.prostate_label_value = prostate_label_value # the integer value of prostate
        self.resize_to = resize_to # tuple of (H, W)

        self.img_dir = os.path.join(data_dir, f"keras_slices_{subset}")
        self.seg_dir = os.path.join(data_dir, f"keras_slices_seg_{subset}")

        self.image_files = sorted(glob.glob(os.path.join(self.img_dir, "*.nii.gz")))
        self.mask_files = sorted(glob.glob(os.path.join(self.seg_dir, "*.nii.gz")))

        # validate the files exist
        if len(self.image_files) == 0:
            raise FileNotFoundError(f"no file in {self.img_dir}")
        if len(self.mask_files) == 0:
            raise FileNotFoundError(f"no mask file in {self.seg_dir}")
        
        # check the number of figure and mask are mactch
        if subset != "validate" and len(self.image_files) != len(self.mask_files):
            print(f"The number of Figure ({len(self.image_files)}) and Mask ({len(self.mask_files)}) are not match")

        print(f"Success load {subset} set: find {len(self.image_files)} figure file。")

    def __len__(self):
        return len(self.image_files) 
    

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        # path of mask
        img_filename = os.path.basename(img_path)
        mask_filename = img_filename.replace("case_", "seg_")
        mask_path = os.path.join(self.seg_dir, mask_filename)

        # back to index mathch
        if not os.path.exists(mask_path):
            if idx < len(self.mask_files):
                mask_path = self.mask_files[idx]
            else:
                raise FileNotFoundError(f"can't find mask for {img_path} figure")

        # load Nifti file
        image = nib.load(img_path).get_fdata().astype(np.float32)
        mask = nib.load(mask_path).get_fdata().astype(np.uint8)

        # change to Tensors
        image_tensor = torch.from_numpy(image.copy()).unsqueeze(0)  # (1,H,W)
        mask_tensor = torch.from_numpy(mask.copy()).long()          # (H,W)

        # Resize
        if self.resize_to:
            image_tensor = TF.resize(image_tensor, self.resize_to, interpolation=TF.InterpolationMode.BILINEAR)
            # unsqueeze, then resize, then squeeze
            mask_tensor = TF.resize(mask_tensor.unsqueeze(0), self.resize_to, interpolation=TF.InterpolationMode.NEAREST).squeeze(0)

        # Z-score normalization
        mean, std = image_tensor.mean(), image_tensor.std()
        # 1e-6 to prevent dividing by 0
        image_tensor = (image_tensor - mean) / (std + 1e-6) 

        # binaray mask
        binary_mask = (mask_tensor == self.prostate_label_value).long()

        return image_tensor, binary_mask


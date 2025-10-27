import torch
from torch.utils.data import Dataset

class HipMRIDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_files = []
        self.mask_files = []
        print("Dataset initialized (basic structure).")

    def __len__(self):

        return 0 

    def __getitem__(self, idx):
        
        return None
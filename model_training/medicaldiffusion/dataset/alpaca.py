import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import nibabel as nib
from scipy.ndimage import zoom


class AlpacaCSVDataset(Dataset):
    def __init__(self, csv_file: str, image_size: int = 64, depth_size: int = 16):
        super().__init__()
        self.df = pd.read_csv(csv_file)
        self.df = self.df.dropna(subset=['mri_path'])
        self.df = self.df[self.df['mri_path'].astype(str).str.strip() != '']
        self.image_size = int(image_size)
        self.depth_size = int(depth_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        mri_path = self.df.iloc[idx]['mri_path']
        img = nib.load(mri_path).get_fdata().astype(np.float32)

        img = (img - np.mean(img)) / (np.std(img) + 1e-8)
        img = np.clip(img, -5, 5)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = img * 2 - 1

        target_shape = (self.depth_size, self.image_size, self.image_size)
        zoom_factors = [target_shape[i] / img.shape[i] for i in range(3)]
        img = zoom(img, zoom_factors, order=1)

        tensor = torch.from_numpy(img).unsqueeze(0)  # (1, D, H, W)
        return {"data": tensor}
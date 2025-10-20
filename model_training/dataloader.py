import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

class MRITabularDataset(Dataset):
    def __init__(self, csv, transform=None, scaler=None):
        self.df = csv
        
        self.df = self.df.dropna(subset=['mri_path'])
        self.df = self.df[self.df['mri_path'].str.strip() != '']
        
        self.tabular_cols = [c for c in self.df.columns if c != "mri_path"]
        self.transform = transform
        self.scaler = scaler

        tabular = self.df[self.tabular_cols].values.astype(np.float32)
        if self.scaler:
            tabular = self.scaler.transform(tabular)
        self.tabular = torch.tensor(tabular, dtype=torch.float32)
        
        print(f"Dataset loaded: {len(self.df)} MRI images and {len(self.tabular_cols)} tabular data")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        mri_path = self.df.iloc[idx]["mri_path"]
        mri = nib.load(mri_path).get_fdata().astype(np.float32)

        mri = (mri - np.mean(mri)) / (np.std(mri) + 1e-8)
        mri = np.clip(mri, -5, 5)
        mri = (mri - mri.min()) / (mri.max() - mri.min())
        mri = mri * 2 - 1

        target_shape = (144, 192, 144)
        zoom_factors = [target_shape[i] / mri.shape[i] for i in range(3)]
        mri = zoom(mri, zoom_factors, order=1)

        mri_tensor = torch.from_numpy(mri).unsqueeze(0)

        if self.transform:
            mri_tensor = self.transform(mri_tensor)

        tabular_data = self.tabular[idx]

        return {
            "mri": mri_tensor,
            "tabular": tabular_data
        }


if __name__ == "__main__":
    dataset = MRITabularDataset(csv_file="model_training/X_train_final.csv")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    for batch in dataloader:
        print("MRI batch:", batch["mri"].shape)
        print("Tabular batch:", batch["tabular"].shape)
        break
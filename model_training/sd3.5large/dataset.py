# dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
from typing import Dict, List, Tuple
import pandas as pd

class MedicalMRIDataset(Dataset):
    """
    Dataset for multi-modal medical MRI data
    Combines 3D MRI scans with tabular clinical data
    """
    def __init__(
        self,
        mri_paths: List[str],
        clinical_data: pd.DataFrame,
        clinical_features: List[str],
        clinical_outcomes: List[str],
        disease_labels: np.ndarray,
        transform=None,
        normalize: bool = True
    ):
        """
        Args:
            mri_paths: List of paths to .nii MRI files
            clinical_data: DataFrame with clinical features
            clinical_features: List of feature column names
            clinical_outcomes: List of outcome column names (ADNI scores)
            disease_labels: Array of disease labels (0=CN, 1=MCI, 2=AD)
            transform: Optional transforms
            normalize: Whether to normalize MRI intensities
        """
        self.mri_paths = mri_paths
        self.clinical_data = clinical_data
        self.clinical_features = clinical_features
        self.clinical_outcomes = clinical_outcomes
        self.disease_labels = disease_labels
        self.transform = transform
        self.normalize = normalize
    
    def __len__(self) -> int:
        return len(self.mri_paths)
    
    def _load_mri(self, path: str) -> torch.Tensor:
        """Load and preprocess MRI scan"""
        # Load NIfTI file
        mri = nib.load(path).get_fdata()
        
        # Normalize intensities
        if self.normalize:
            mri = (mri - mri.mean()) / (mri.std() + 1e-8)
            mri = np.clip(mri, -5, 5)  # Clip outliers
        
        # Add channel dimension
        mri = mri[np.newaxis, ...]  # (1, D, H, W)
        
        return torch.FloatTensor(mri)
    
    def _get_clinical_data(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get clinical features and outcomes"""
        row = self.clinical_data.iloc[idx]
        
        # Extract features
        features = row[self.clinical_features].values.astype(np.float32)
        features = torch.FloatTensor(features)
        
        # Extract outcomes
        outcomes = row[self.clinical_outcomes].values.astype(np.float32)
        outcomes = torch.FloatTensor(outcomes)
        
        return features, outcomes
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single data sample"""
        # Load MRI
        mri = self._load_mri(self.mri_paths[idx])
        
        # Get clinical data
        clinical_features, clinical_outcomes = self._get_clinical_data(idx)
        
        # Get disease label
        disease_label = torch.LongTensor([self.disease_labels[idx]])[0]
        
        sample = {
            'mri': mri,
            'tabular': clinical_features,
            'clinical_outcomes': clinical_outcomes,
            'disease_labels': disease_label
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


def create_dataloaders(
    train_mri_paths: List[str],
    val_mri_paths: List[str],
    train_clinical_data: pd.DataFrame,
    val_clinical_data: pd.DataFrame,
    clinical_features: List[str],
    clinical_outcomes: List[str],
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    batch_size: int = 8,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders"""
    from torch.utils.data import DataLoader
    
    train_dataset = MedicalMRIDataset(
        mri_paths=train_mri_paths,
        clinical_data=train_clinical_data,
        clinical_features=clinical_features,
        clinical_outcomes=clinical_outcomes,
        disease_labels=train_labels,
        normalize=True
    )
    
    val_dataset = MedicalMRIDataset(
        mri_paths=val_mri_paths,
        clinical_data=val_clinical_data,
        clinical_features=clinical_features,
        clinical_outcomes=clinical_outcomes,
        disease_labels=val_labels,
        normalize=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

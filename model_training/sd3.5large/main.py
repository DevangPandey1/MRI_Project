# main.py
import torch
import pandas as pd
import numpy as np
from pathlib import Path

from medical_sd35_model import MedicalSD35MultiModal
from train import MultiTaskTrainer
from dataset import create_dataloaders

def main():
    # Configuration
    config = {
        'sd35_model_path': "stabilityai/stable-diffusion-3.5-large",
        'num_clinical_features': 20,  # Adjust to your ADNI features
        'num_clinical_outcomes': 5,    # ADNI_MEM, ADNI_EF, etc.
        'num_disease_classes': 3,      # CN, MCI, AD
        'batch_size': 4,               # Adjust based on GPU memory
        'learning_rate': 1e-4,
        'num_epochs': 50,
        'unfreeze_sd35_epoch': 20,     # Unfreeze SD3.5 after 20 epochs
        'use_quantization': True,      # Use 4-bit quantization to reduce VRAM
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("Loading data...")

    train_csv = Path("../../X_train.csv")
    val_csv = Path("../../X_val.csv")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    # Load your data (adjust paths as needed)
    train_mri_paths = train_df['mri_path'].tolist()
    val_mri_paths = val_df['mri_path'].tolist()
    
    # Load clinical data
    train_clinical = pd.read_csv("data/train/clinical_data.csv")
    val_clinical = pd.read_csv("data/val/clinical_data.csv")
    
    # Define feature columns
    clinical_features = [
        'AGE', 'PTGENDER', 'PTEDUCAT', 'APOE4',
        # Add your other clinical features
    ]
    clinical_outcomes = [
        'ADNI_MEM', 'ADNI_EF', 'ADNI_LAN', 'ADNI_VISSPAT', 'ADNI_VS'
    ]
    
    # Load disease labels
    train_labels = train_clinical['DX'].map({'CN': 0, 'MCI': 1, 'AD': 2}).values
    val_labels = val_clinical['DX'].map({'CN': 0, 'MCI': 1, 'AD': 2}).values
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_mri_paths=train_mri_paths,
        val_mri_paths=val_mri_paths,
        train_clinical_data=train_clinical,
        val_clinical_data=val_clinical,
        clinical_features=clinical_features,
        clinical_outcomes=clinical_outcomes,
        train_labels=train_labels,
        val_labels=val_labels,
        batch_size=config['batch_size'],
        num_workers=4
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Initialize model
    print("Initializing Medical SD3.5 model...")
    model = MedicalSD35MultiModal(
        sd35_model_path=config['sd35_model_path'],
        num_clinical_features=config['num_clinical_features'],
        num_clinical_outcomes=config['num_clinical_outcomes'],
        num_disease_classes=config['num_disease_classes'],
        freeze_sd35_initially=True,
        use_quantization=config['use_quantization']
    )
    
    print(f"Model initialized on {config['device']}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Initialize trainer
    trainer = MultiTaskTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config['device'],
        learning_rate=config['learning_rate'],
        task_weights={
            'clinical': 1.0,
            'classification': 1.0,
            'reconstruction': 0.3  # Lower weight for reconstruction
        },
        use_mixed_precision=True,
        log_wandb=True
    )
    
    # Train model
    print("\nStarting training...")
    trainer.train(
        num_epochs=config['num_epochs'],
        unfreeze_sd35_epoch=config['unfreeze_sd35_epoch']
    )
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()

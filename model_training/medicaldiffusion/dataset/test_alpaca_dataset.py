#!/usr/bin/env python3

import sys
import os

# Add the parent directory to Python path so we can import from the parent module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from dataset.alpaca import AlpacaCSVDataset
import torch

def test_alpaca_dataset():    
    csv_file = '/Users/rishipc/Documents/Studies/Fall 2025/Independent Study/ALPACA-3D/model_training/X_train.csv'
    
    print("Testing ALPACA dataset loading...")
    print(f"CSV file: {csv_file}")
    
    try:
        dataset = AlpacaCSVDataset(
            csv_file=csv_file,
            image_size=64,
            depth_size=16
        )
        
        print(f"Dataset created successfully!")
        print(f"Total samples: {len(dataset)}")
        
        if len(dataset) > 0:
            for i in range(min(3, len(dataset))):
                sample = dataset[i]
                print(f"Sample {i}: shape {sample['data'].shape}")
        
        print("\nAll tests passed!")
        return True
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False

if __name__ == "__main__":
    test_alpaca_dataset()
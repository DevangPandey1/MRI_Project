import os
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import pyvista as pv

class ADNI_Dataset(Dataset):
    def __init__(self, adni_root, target_shape=(128, 128, 128), transform=None):
        self.nii_paths = []
        
        for subj in os.listdir(adni_root):
            subj_path = os.path.join(adni_root, subj)
            if not os.path.isdir(subj_path):
                continue

            scan_folders = sorted([d for d in os.listdir(subj_path) if os.path.isdir(os.path.join(subj_path, d))])
            if not scan_folders:
                continue
            first_scan = os.path.join(subj_path, scan_folders[0])
                
            for date_folder in os.listdir(first_scan):
                date_path = os.path.join(first_scan, date_folder)
                if not os.path.isdir(date_path):
                    continue
                    
                for img_id in os.listdir(date_path):
                    img_id_path = os.path.join(date_path, img_id)
                    if not os.path.isdir(img_id_path):
                        continue
                        
                    for file in os.listdir(img_id_path):
                        if file.endswith('.nii'):
                            full_path = os.path.join(img_id_path, file)
                            self.nii_paths.append(full_path)

        self.target_shape = target_shape
        self.transform = transform
        print(f"Found {len(self.nii_paths)} MRI volumes")

    def __len__(self):
        return len(self.nii_paths)

    def __getitem__(self, idx):
        nii_path = self.nii_paths[idx]
        
        img = nib.load(nii_path).get_fdata().astype(np.float32)
        
        if np.max(img) > np.min(img):
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
        
        zoom_factors = [self.target_shape[i] / img.shape[i] for i in range(3)]
        img_resized = zoom(img, zoom_factors, order=1)
        
        img_resized = np.expand_dims(img_resized, axis=0)

        tensor_img = torch.tensor(img_resized, dtype=torch.float32)
        if self.transform:
            tensor_img = self.transform(tensor_img)
        return tensor_img

def show_mri_volume_3d(volume, num_slices=3, save_path=None, title="3D MRI Volume"):
    if isinstance(volume, torch.Tensor):
        volume = volume.squeeze().cpu().numpy()
    
    if len(volume.shape) != 3:
        raise ValueError(f"Expected 3D volume, got shape {volume.shape}")
    
    def get_slice_indices(total_slices, num_slices):
        return [int(i * total_slices / (num_slices + 1)) for i in range(1, num_slices + 1)]
    
    axial_indices = get_slice_indices(volume.shape[2], num_slices)
    coronal_indices = get_slice_indices(volume.shape[1], num_slices)  
    sagittal_indices = get_slice_indices(volume.shape[0], num_slices)
    
    fig, axes = plt.subplots(3, num_slices, figsize=(15, 12))
    fig.suptitle(title, fontsize=16)
    
    for i, idx in enumerate(axial_indices):
        axes[0, i].imshow(volume[:, :, idx].T, cmap='gray', origin='lower')
        axes[0, i].set_title(f'Axial Slice {idx}')
        axes[0, i].axis('off')
    
    for i, idx in enumerate(coronal_indices):
        axes[1, i].imshow(volume[:, idx, :].T, cmap='gray', origin='lower')
        axes[1, i].set_title(f'Coronal Slice {idx}')
        axes[1, i].axis('off')
    
    for i, idx in enumerate(sagittal_indices):
        axes[2, i].imshow(volume[idx, :, :].T, cmap='gray', origin='lower')
        axes[2, i].set_title(f'Sagittal Slice {idx}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved 3D visualization to {save_path}')
    plt.close()

def visualize_internal_volume(volume, title="Internal MRI Volume"):
    if isinstance(volume, torch.Tensor):
        volume = volume.squeeze().cpu().numpy()
    
    grid = pv.ImageData()
    grid.dimensions = volume.shape
    grid.spacing = (1, 1, 1)
    grid.point_data['scalars'] = volume.flatten(order='F')
    
    plotter = pv.Plotter()
    plotter.add_text(title, font_size=12)
    
    plotter.add_volume(grid, 
                      opacity='sigmoid',
                      cmap='gray',
                      opacity_unit_distance=0.3,
                      shade=True)
    
    plotter.show_axes()
    plotter.set_background('black')
    plotter.show()

def save_paraview_file(volume, base_name="mri_volume"):
    
    if isinstance(volume, torch.Tensor):
        volume = volume.squeeze().cpu().numpy()
    
    grid = pv.ImageData()
    grid.dimensions = volume.shape
    grid.spacing = (1, 1, 1)
    grid.point_data['scalars'] = volume.flatten(order='F')
    
    vtk_path = f"{base_name}.vtk"
    grid.save(vtk_path)
    print(f"Saved VTK file: {vtk_path}")
    
    return vtk_path


if __name__ == "__main__":
    adni_root = 'ADNI'
    
    dataset = ADNI_Dataset(adni_root, target_shape=(128, 128, 128))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    print(f"\nGenerating 3D Visualizations")
    
    random_idx = np.random.randint(0, len(dataset))
    vol = dataset[random_idx]
    vol = vol.unsqueeze(0)
    
    print(f"\nProcessing Volume...")
    
    file_path = f'adni_3d_mri_volume.png'
    show_mri_volume_3d(vol[0], num_slices=3, save_path=file_path, title=f"ADNI 3D MRI Volume")
    
    base_name = f'adni_volume'
    vtk_path = save_paraview_file(vol[0], base_name)
    
    print(f"Opening internal volume visualization")
    visualize_internal_volume(vol[0], title=f"ADNI Internal Volume")

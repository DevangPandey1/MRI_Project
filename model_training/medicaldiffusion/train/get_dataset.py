from dataset import ADNIDataset, AlpacaCSVDataset, DEFAULTDataset


def get_dataset(cfg):
    if cfg.dataset.name == 'ALPACA_CSV':
        train_dataset = AlpacaCSVDataset(
            csv_file=cfg.dataset.csv_file,
            image_size=cfg.model.diffusion_img_size,
            depth_size=cfg.model.diffusion_depth_size
        )
        val_dataset = AlpacaCSVDataset(
            csv_file=cfg.dataset.csv_file,
            image_size=cfg.model.diffusion_img_size,
            depth_size=cfg.model.diffusion_depth_size
        )
        sampler = None
        return train_dataset, val_dataset, sampler

    if cfg.dataset.name == 'ADNI':
        train_dataset = ADNIDataset(
            root_dir=cfg.dataset.root_dir, augmentation=True)
        val_dataset = ADNIDataset(
            root_dir=cfg.dataset.root_dir, augmentation=True)
        sampler = None
        return train_dataset, val_dataset, sampler
        
    if cfg.dataset.name == 'DEFAULT':
        train_dataset = DEFAULTDataset(
            root_dir=cfg.dataset.root_dir)
        val_dataset = DEFAULTDataset(
            root_dir=cfg.dataset.root_dir)
        sampler = None
        return train_dataset, val_dataset, sampler
    
    raise ValueError(f'{cfg.dataset.name} Dataset is not available')

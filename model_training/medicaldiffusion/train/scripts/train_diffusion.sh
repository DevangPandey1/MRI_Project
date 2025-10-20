#!/bin/bash

source "/data/home/firas/anaconda3/etc/profile.d/conda.sh"
conda activate vq_gan_3d
export PYTHONPATH=$PWD

 # ADNI
python train/train_ddpm.py model=ddpm dataset=adni model.vqgan_ckpt='/data/home/firas/Desktop/work/other_groups/medicaldiffusion/checkpoints/vq_gan/ADNI/roi/lightning_logs/version_1/checkpoints/epoch\=99-step\=99000-train/recon_loss\=0.05.ckpt' model.diffusion_img_size=32 model.diffusion_depth_size=32 model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.batch_size=10 model.gpus=2 model.results_folder_postfix='roi'
python train/train_ddpm.py model=ddpm dataset=adni model.vqgan_ckpt='/data/home/firas/Desktop/work/other_groups/medicaldiffusion/checkpoints/vq_gan/ADNI/roi/lightning_logs/version_1/checkpoints/epoch\=99-step\=99000-train/recon_loss\=0.05.ckpt' model.diffusion_img_size=32 model.diffusion_depth_size=32 model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.batch_size=10 model.gpus=3 model.results_folder_postfix='roi_normal_unet' model.denoising_fn='UNet'

 # ALPACA CSV
 # python train/train_ddpm.py model=ddpm dataset=alpaca model.vqgan_ckpt='<PATH_TO_VQGAN_CKPT>' model.diffusion_img_size=32 model.diffusion_depth_size=32 model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.batch_size=10 model.gpus=1
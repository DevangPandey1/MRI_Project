# Medical Diffusion

This repository contains the code to our paper "Medical Diffusion: Denoising Diffusion Probabilistic Models for 3D Medical Image Synthesis"
(see https://arxiv.org/abs/2211.03364).

![Generated Samples by our Medical Diffusion model](assets/generated_samples.gif)

# System Requirements
This code has been tested on Ubuntu 20.04 and an NVIDIA Quadro RTX 6000 GPU. Furthermore it was developed using Python v3.8.

# Setup
In order to run our model, we suggest you create a virtual environment 
```
conda create -n medicaldiffusion python=3.8
``` 
and activate it with 
```
conda activate medicaldiffusion
```
Subsequently, download and install the required libraries by running 
```
pip install -r requirements.txt
```

# Training
Once all libraries are installed and the datasets have been downloaded, you are ready to train the model.

We currently support training on ADNI and ALPACA CSV.

ADNI VQ-GAN training example:
```
PL_TORCH_DISTRIBUTED_BACKEND=gloo python train/train_vqgan.py dataset=adni dataset.root_dir=<INSERT_PATH_TO_ADNI_DATASET> model=vq_gan_3d model.gpus=1 model.precision=16 model.embedding_dim=8 model.n_hiddens=16 model.downsample=[2,2,2] model.num_workers=32 model.gradient_clip_val=1.0 model.lr=3e-4 model.discriminator_iter_start=10000 model.perceptual_weight=4 model.image_gan_weight=1 model.video_gan_weight=1 model.gan_feat_weight=4 model.batch_size=2 model.n_codes=16384 model.accumulate_grad_batches=1 
```

ADNI DDPM training example:
```
python train/train_ddpm.py model=ddpm dataset=adni model.vqgan_ckpt=<INSERT_PATH_TO_CHECKPOINT> model.diffusion_img_size=32 model.diffusion_depth_size=32 model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.batch_size=10 model.gpus=1
```

ALPACA CSV examples (assuming an `ALPACA_CSV` dataset with a CSV path configured in the hydra config):
```
PL_TORCH_DISTRIBUTED_BACKEND=gloo python train/train_vqgan.py dataset=alpaca model=vq_gan_3d model.gpus=1
python train/train_ddpm.py model=ddpm dataset=alpaca model.vqgan_ckpt=<INSERT_PATH_TO_CHECKPOINT> model.gpus=1
```


# Citation
To cite our work, please use
```
@misc{https://doi.org/10.48550/arxiv.2211.03364,
  doi = {10.48550/ARXIV.2211.03364},
  url = {https://arxiv.org/abs/2211.03364},
  author = {Khader, Firas and Mueller-Franzes, Gustav and Arasteh, Soroosh Tayebi and Han, Tianyu and Haarburger, Christoph and Schulze-Hagen, Maximilian and Schad, Philipp and Engelhardt, Sandy and Baessler, Bettina and Foersch, Sebastian and Stegmaier, Johannes and Kuhl, Christiane and Nebelung, Sven and Kather, Jakob Nikolas and Truhn, Daniel},
  title = {Medical Diffusion - Denoising Diffusion Probabilistic Models for 3D Medical Image Generation},
  publisher = {arXiv},
  year = {2022},
}
```


# Acknowledgement
This code is heavily build on the following repositories:

(1) https://github.com/SongweiGe/TATS

(2) https://github.com/lucidrains/denoising-diffusion-pytorch

(3) https://github.com/lucidrains/video-diffusion-pytorch

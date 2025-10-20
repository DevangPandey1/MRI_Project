#!/bin/bash
#SBATCH --job-name=alpaca_vqgan
#SBATCH --account=ucb-general          # REQUIRED on CURC
#SBATCH --partition=aa100        # GPU partition for training
#SBATCH --qos=normal                  # Standard QoS on Alpine
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8             # CPU cores for data loading
#SBATCH --mem=64G                     # More memory for training
#SBATCH --time=12:00:00              # 12 hours walltime
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --output=model_training/medicaldiffusion/train/vqgan_training.out
#SBATCH --error=model_training/medicaldiffusion/train/vqgan_training.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nobr3541@colorado.edu,ripa7320@colorado.edu

# Make the output files group files
umask 002

# Environment setup
module purge
module load python/3.8.10
module load cuda/11.8.0

# Activate your virtual environment
source /scratch/alpine/nobr3541/.venv/bin/activate

# Set Python path
export PYTHONPATH=$PWD

# Navigate to medicaldiffusion directory
cd /projects/nobr3541/ALPACA-3D/model_training/medicaldiffusion

# Create checkpoints directory
mkdir -p checkpoints/vq_gan/ALPACA_CSV

# Train VQ-GAN model
echo "Starting VQ-GAN training for ALPACA dataset"
echo "Job started at: $(date)"

PL_TORCH_DISTRIBUTED_BACKEND=gloo python train/train_vqgan.py \
    dataset=alpaca \
    model=vq_gan_3d \
    model.gpus=1 \
    model.default_root_dir_postfix='alpaca_training' \
    model.precision=16 \
    model.embedding_dim=8 \
    model.n_hiddens=16 \
    model.downsample=[2,2,2] \
    model.num_workers=8 \
    model.gradient_clip_val=1.0 \
    model.lr=3e-4 \
    model.discriminator_iter_start=10000 \
    model.perceptual_weight=4 \
    model.image_gan_weight=1 \
    model.video_gan_weight=1 \
    model.gan_feat_weight=4 \
    model.batch_size=2 \
    model.n_codes=16384 \
    model.accumulate_grad_batches=1

echo "VQ-GAN training completed at: $(date)"
echo "Checkpoint saved in: checkpoints/vq_gan/ALPACA_CSV/alpaca_training/"

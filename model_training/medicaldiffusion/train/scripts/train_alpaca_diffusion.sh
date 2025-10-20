#!/bin/bash
#SBATCH --job-name=alpaca_diffusion
#SBATCH --account=ucb-general          # REQUIRED on CURC
#SBATCH --partition=amilan-gpu        # GPU partition for training
#SBATCH --qos=normal                  # Standard QoS on Alpine
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8             # CPU cores for data loading
#SBATCH --mem=64G                     # More memory for training
#SBATCH --time=24:00:00              # 24 hours walltime (diffusion takes longer)
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --output=diffusion_training.out
#SBATCH --error=diffusion_training.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nobr3541@colorado.edu

# Make the output files group files
umask 002

# --- Environment setup ---
module purge
module load python/3.8.10
module load cuda/11.8.0

# Activate your virtual environment (adjust path as needed)
source /scratch/alpine/nobr3541/.venv/bin/activate

# Set Python path
export PYTHONPATH=$PWD

# --- Navigate to medicaldiffusion directory ---
cd /scratch/alpine/nobr3541/ALPACA-3D/model_training/medicaldiffusion

# Create checkpoints directory
mkdir -p checkpoints/ddpm/ALPACA_CSV

# --- Train DDPM model ---
echo "Starting DDPM training for ALPACA dataset..."
echo "Job started at: $(date)"

# Note: Update the vqgan_ckpt path to your actual VQ-GAN checkpoint
python train/train_ddpm.py \
    model=ddpm \
    dataset=alpaca \
    model.results_folder_postfix='alpaca_diffusion' \
    model.vqgan_ckpt='checkpoints/vq_gan/ALPACA_CSV/alpaca_training/lightning_logs/version_0/checkpoints/latest_checkpoint.ckpt' \
    model.diffusion_img_size=32 \
    model.diffusion_depth_size=32 \
    model.diffusion_num_channels=8 \
    model.dim_mults=[1,2,4,8] \
    model.batch_size=10 \
    model.gpus=1

echo "DDPM training completed at: $(date)"
echo "Results saved in: checkpoints/ddpm/ALPACA_CSV/alpaca_diffusion/"

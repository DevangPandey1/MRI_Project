#!/bin/bash
#SBATCH --job-name=medical_sd35
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --account=ucb-general  # Replace with your allocation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=1:00:00
#SBATCH --output=logs/medical_sd35_%j.out
#SBATCH --error=logs/medical_sd35_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL      # Events to notify on (job starts, ends, or fails)
#SBATCH --mail-user=depa9289@colorado.edu   # Your CU email

# Load necessary modules
module purge
module load slurm/alpine
module load anaconda

# Activate your conda environment (pre-created with dependencies)
conda activate your_env_name

# Create logs directory
mkdir -p logs

# Run training
python main.py
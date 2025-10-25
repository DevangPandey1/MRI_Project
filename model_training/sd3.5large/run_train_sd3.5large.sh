#!/bin/bash
#SBATCH --job-name=medical_sd35
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --account=ucb-general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=1:00:00
#SBATCH --output=logs/medical_sd35_%j.out
#SBATCH --error=logs/medical_sd35_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=depa9289@colorado.edu

# Load necessary modules
module purge
module load slurm/alpine
module load anaconda

# Activate your conda environment (replace with your actual env name)
conda activate medical_sd

# Create logs directory if it doesn't exist
mkdir -p logs

# Run training
python main.py

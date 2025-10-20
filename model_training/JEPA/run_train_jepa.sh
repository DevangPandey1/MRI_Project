#!/bin/bash
#SBATCH --job-name=jepa_encoder
#SBATCH --account=ucb698_asc1          # REQUIRED on CURC (e.g., ucb-general for random stuff, and ucb698_asc1 for the GPU)
#SBATCH --partition=aa100             # Alpine CPU partition
#SBATCH --qos=long 		      # Standard QoS on Alpine (normal, long)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8              # adjust as needed
#SBATCH --mem=50GB                       # adjust as needed
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00                # walltime (DD-HH:MM:SS)
#SBATCH --output=train_jepa.out
#SBATCH --error=train_jepa.err
#SBATCH --mail-type=BEGIN,END,FAIL      # Events to notify on (job starts, ends, or fails)
#SBATCH --mail-user=nobr3541@colorado.edu   # Your CU email

# Make the output files group files
umask 002

# --- Environment setup (modules/conda) ---
module purge
# Activate local env
acompile
module load anaconda
conda activate alpaca_3d

# --- Your command ---
python3 train_jepa.py

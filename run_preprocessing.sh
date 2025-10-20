#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --account=ucb-general          # REQUIRED on CURC (e.g., ucb-general, your lab's account)
#SBATCH --partition=amilan             # Alpine CPU partition
#SBATCH --qos=normal                   # Standard QoS on Alpine
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8              # adjust as needed
#SBATCH --mem=32G                       # adjust as needed
#SBATCH --time=01:00:00                # walltime (HH:MM:SS)
#SBATCH --output=preprocessing.out
#SBATCH --error=preprocessing.err
#SBATCH --mail-type=BEGIN,END,FAIL      # Events to notify on (job starts, ends, or fails)
#SBATCH --mail-user=nobr3541@colorado.edu   # Your CU email

# Make the output files group files
umask 002

# --- Environment setup (modules/conda) ---
module purge
# Activate local env
source .venv/bin/activate

# --- Run from the submission directory (default is $SLURM_SUBMIT_DIR) ---
SLURM_SUBMIT_DIR="./preprocessing"
cd "$SLURM_SUBMIT_DIR"

# --- Your command ---
python preprocess.py

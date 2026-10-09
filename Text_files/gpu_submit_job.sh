#!/bin/bash
#SBATCH --job-name=power_law_scan
#SBATCH --partition=gpu
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# 1. Load system Anaconda/Miniconda module (if required by your cluster)
module load miniconda3-2025.11.1

# 2. Source the Conda profile script to enable 'conda activate' in non-interactive batch mode
source $(conda info --base)/etc/profile.d/conda.sh

# 3. Activate your environment created in /home/data/
conda activate /home/data/sambuddha_fac/divesh/gpu_env

# 4. Optional: Print GPU info to the log file to confirm GPU allocation
nvidia-smi

# 5. Run your Python script
python $1
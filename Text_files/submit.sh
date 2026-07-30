#!/bin/bash
#SBATCH --job-name=RMP_Linear_Scan
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64          # This value is what Python will now see
#SBATCH --partition=short
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --mem=128G

# 1. Load your Conda environment
module load miniconda3-2025.11.1
source activate quantum_env

# 2. Run the script 
# No 'export' needed here because you updated the .py file!
python Heisenberg_1DNN_Cvh.py

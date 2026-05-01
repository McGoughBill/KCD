#!/bin/bash

#SBATCH --account YOUR_SLURM_ACCOUNT
#SBATCH --qos epsrc
#SBATCH --time 0-06:00:00
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --cpus-per-gpu 48
#SBATCH --mem 200G

set -x

module purge; module load baskerville
module load bask-apps/test
module load Miniconda3/4.10.3
conda init bash
source ${HOME}/.bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/kcd_data/conda_env/ovseg_env/lib

# Location of conda environment
CONDA_ENV_PATH="/path/to/kcd_data/conda_env/ovseg_env"

# Activate the environment
conda activate "${CONDA_ENV_PATH}"


# Run jobs

# python KCD/Segmentation/Inference/End2End.py
python KCD/Segmentation/Inference/Postprocess_Testing.py

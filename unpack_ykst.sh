#!/bin/bash

#SBATCH --account YOUR_SLURM_ACCOUNT
#SBATCH --qos epsrc
#SBATCH --time 0-48:00:00
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --cpus-per-gpu 48
#SBATCH --mem 400G

set -x

module purge; module load baskerville
module load bask-apps/test
module load Miniconda3/4.10.3
conda init bash
source ${HOME}/.bashrc

# Location of conda environment
CONDA_ENV_PATH="/path/to/kcd_data/conda_env/ovseg_env"

# Activate the environment
conda activate "${CONDA_ENV_PATH}"


# Run jobs
python KCD/utils/dcm2nii.py 
#!/bin/bash -l
#SBATCH --job-name=examplejob   # Job name
#SBATCH --output=/scratch/project_465002373/protbert/output.txt  # Name of stdout output file
#SBATCH --error=/scratch/project_465002373/protbert/error.txt # Name of stderr error file
#SBATCH --partition=small-g       # Partition name
#SBATCH --ntasks=1              # One task (process)
#SBATCH --time=00:15:00         # Run time (hh:mm:ss)
#SBATCH --account=project_465002373  # Project for billing
#SBATCH --gpus-per-node=1

export PROJECT_DIR=/scratch/project_465002373/protbert
export MNT_DIR_CONTAINER=/mnt/data
export WANDB_DIR=${MNT_DIR_CONTAINER}/wandb

export CONTAINER=${PROJECT_DIR}/rocm-pytorch-wand-composer.sif

singularity exec  -B ${PROJECT_DIR}:${MNT_DIR_CONTAINER} --env WANDB_DIR=${WANDB_DIR}  $CONTAINER python3 ${MNT_DIR_CONTAINER}/train_single.py  --base_dir=${MNT_DIR_CONTAINER}/datasets/ --datasets_prefix="dataset_255w_homology_split_" --epochs 5 --batch_size 56

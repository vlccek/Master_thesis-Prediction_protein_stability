#!/bin/bash
#SBATCH --job-name=protbert_train_composer
#SBATCH --account=project_465002373
#SBATCH --partition=small-g       # Partition name
#SBATCH --gpus-per-node=8
#SBATCH --time=16:00:00       # Časový limit
#SBATCH --mem=256G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Nastavení proměnných (pokud nejsou nastaveny v .bashrc)
export PROJECT_DIR=/flash/project_465002373/protbert
export MNT_DIR_CONTAINER=/mnt/data
export WANDB_DIR=${MNT_DIR_CONTAINER}/wandb

export CONTAINER=${PROJECT_DIR}/rocm-protbert-wand-composer-rocm7.1.1.sif

# Zavedení singularity modulu (pokud je na clusteru potřeba)
# module load singularity

echo "Spouštím trénink na uzlu: $(hostname)"
echo "Dostupná GPU zařízení: $CUDA_VISIBLE_DEVICES"

# Vlastní spuštění skriptu
# DŮLEŽITÉ: Zde musí být --rocm, aby kontejner viděl GPU

srun singularity exec --rocm -B ${PROJECT_DIR}:${MNT_DIR_CONTAINER} \
    --env WANDB_DIR=${WANDB_DIR} \
    $CONTAINER \
    rocm-smi

srun singularity exec --rocm -B ${PROJECT_DIR}:${MNT_DIR_CONTAINER} \
    --env WANDB_DIR=${WANDB_DIR} \
    $CONTAINER \
    composer -n 8 ${MNT_DIR_CONTAINER}/train_composer.py \
    --epochs 5 \
    --batch_size 56 \
    --base_dir=/mnt/data/datasets/ \
    --datasets_prefix="dataset_255w_"
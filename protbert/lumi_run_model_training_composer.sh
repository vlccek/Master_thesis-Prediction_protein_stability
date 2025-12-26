#!/bin/bash
#SBATCH --job-name=protbert_train_composer
#SBATCH --account=project_465002373
#SBATCH --partition=small-g       # Partition name
#SBATCH --gpus-per-node=8
#SBATCH --time=24:15:00       # Časový limit
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


srun singularity exec --rocm -B ${PROJECT_DIR}:${MNT_DIR_CONTAINER} \
    --env WANDB_DIR=${WANDB_DIR} \
    $CONTAINER \
    bash -c "echo '--- SYSTEM INFO ---'; \
             rocm-smi --showdriverversion; \
             echo '--- PYTHON & PYTORCH ---'; \
             python3 -c \"import torch; import sys; \
             print(f'Python: {sys.version.split()[0]}'); \
             print(f'PyTorch: {torch.__version__}'); \
             print(f'ROCm (HIP) build: {torch.version.hip}'); \
             print(f'CUDA/ROCm available: {torch.cuda.is_available()}'); \
             print(f'Device count: {torch.cuda.device_count()}'); \
             print(f'Current device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}')\""

srun singularity exec --rocm -B ${PROJECT_DIR}:${MNT_DIR_CONTAINER} \
    --env WANDB_DIR=${WANDB_DIR} \
    $CONTAINER \
    composer ${MNT_DIR_CONTAINER}/train_composer.py \
    --epochs 5 \
    --batch_size 40 \
    --base_dir=/mnt/data/datasets/ \
    --datasets_prefix="dataset_homology_split_" \
    --project_name="protein-mutation-prediction-protbert-composer"  \
    --model_name="Rostlab/prot_bert_bfd" \
    --seq_window_size 255


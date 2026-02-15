#!/bin/bash
#SBATCH --job-name=protbert_train_composer
#SBATCH --account=project_465002373
#SBATCH --partition=standard-g      # Partition name
#SBATCH --gpus-per-node=8
#SBATCH --time=24:15:00       # Časový limit
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Nastavení proměnných (pokud nejsou nastaveny v .bashrc)
export PROJECT_DIR=/flash/project_465002373/protbert
export MNT_DIR_CONTAINER=/mnt/data

# --- Time-based Directory Setup ---
# 1. Vygenerujeme timestamp
export START_TIME=$(date +"%Y-%m-%d_%H-%M-%S")
echo "Training Run Start Time: $START_TIME"

export HOST_RUN_DIR="${PROJECT_DIR}/runs/${START_TIME}"
mkdir -p "${HOST_RUN_DIR}"
echo "Created run directory: ${HOST_RUN_DIR}"

# 3. Cesta uvnitř KONTEJNERU (přes bind mount /mnt/data)
# /mnt/data mapuje ${PROJECT_DIR}, takže:
export CONT_RUN_DIR="${MNT_DIR_CONTAINER}/runs/${START_TIME}"

# 4. Nastavení cest pro WandB a Checkpointy do této složky
export WANDB_DIR="${CONT_RUN_DIR}/wandb"
export WANDB_CACHE_DIR="${CONT_RUN_DIR}/wandb_cache"
export WANDB_DATA_DIR="${CONT_RUN_DIR}/wandb_data"
# Absolutní cesta k checkpointům (Python os.path.join ji použije jako absolutní a ignoruje base_dir)
export CHECKPOINT_SAVE_FOLDER="${CONT_RUN_DIR}/checkpoints"

# --- Training Configuration ---
export EPOCHS_FULL=5
export BATCH_SIZE=47
export BASE_DIR="/mnt/data/datasets/"
# export DATASETS_PREFIX="dataset_split_"
export DATASETS_PREFIX="dataset_homology_split_"
export PROJECT_NAME="protein-mutation-prediction-protbert-composer"
export MODEL_NAME="Rostlab/prot_bert_bfd"
export SEQ_WINDOW_SIZE=255
export FREEZED_LAYERS=3

export CPU_BIND_MASKS="0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000"

# Tell RCCL to use Slingshot interfaces and GPU RDMA
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB
# ------------------------------

export CONTAINER=${PROJECT_DIR}/rocm-protbert-wand-composer-rocm7.1.1.sif

echo "Spouštím trénink na uzlu: $(hostname)"
echo "Dostupná GPU zařízení: $CUDA_VISIBLE_DEVICES"


srun --cpu-bind=v,mask_cpu:$CPU_BIND_MASKS singularity exec --rocm -B ${PROJECT_DIR}:${MNT_DIR_CONTAINER} \
    --env WANDB_DIR=${WANDB_DIR} \
    --env WANDB_CACHE_DIR=${WANDB_CACHE_DIR} \
    --env WANDB_DATA_DIR=${WANDB_DATA_DIR} \
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


printf "Starting training script \n \n:"
srun --cpu-bind=v,mask_cpu:$CPU_BIND_MASKS singularity exec --rocm -B ${PROJECT_DIR}:${MNT_DIR_CONTAINER} \
    --env WANDB_DIR=${WANDB_DIR} \
    --env WANDB_CACHE_DIR=${WANDB_CACHE_DIR} \
    --env WANDB_DATA_DIR=${WANDB_DATA_DIR} \
    $CONTAINER \
    composer ${MNT_DIR_CONTAINER}/train_composer.py \
    --batch_size ${BATCH_SIZE} \
    --base_dir="${BASE_DIR}" \
    --datasets_prefix="${DATASETS_PREFIX}" \
    --project_name="${PROJECT_NAME}"  \
    --model_name="${MODEL_NAME}" \
    --seq_window_size ${SEQ_WINDOW_SIZE} \
    --freezed_layers ${FREEZED_LAYERS} \
    --epochs ${EPOCHS_FULL} \
    --save_folder "${CHECKPOINT_SAVE_FOLDER}" \


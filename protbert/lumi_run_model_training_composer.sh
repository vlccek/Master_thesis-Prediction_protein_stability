#!/bin/bash
#SBATCH --job-name=protbert_train_composer
#SBATCH --account=project_465002740
#SBATCH --partition=standard-g      # Partition name
#SBATCH --gpus-per-node=8
#SBATCH --time=24:15:00       # Časový limit
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Nastavení proměnných (pokud nejsou nastaveny v .bashrc)
export PROJECT_DIR=/flash/project_465002740/protbert
export MNT_DIR_CONTAINER=/mnt/data

# --- Time-based Directory Setup ---
# 1. Vygenerujeme timestamp
export START_TIME=$(date +"%Y-%m-%d_%H-%M-%S")
echo "Training Run Start Time: $START_TIME"

export HOST_RUN_DIR="${PROJECT_DIR}/runs/${START_TIME}"
mkdir -p "${HOST_RUN_DIR}/logs"
echo "Created run directory: ${HOST_RUN_DIR}"

# Update SBATCH output/error paths dynamically if possible, 
# but since sbatch already parsed them, we'll rely on the user running from project root or 
# we can't easily change it once started. 
# HOWEVER, we can make sure the 'logs' folder exists in PROJECT_DIR.

mkdir -p "${PROJECT_DIR}/logs"

# 3. Cesta uvnitř KONTEJNERU (přes bind mount /mnt/data)
# /mnt/data mapuje ${PROJECT_DIR}, takže:
export CONT_RUN_DIR="${MNT_DIR_CONTAINER}/runs/${START_TIME}"

# 4. Nastavení cest pro WandB a Checkpointy do této složky
export WANDB_DIR="${CONT_RUN_DIR}/wandb"
export WANDB_CACHE_DIR="${CONT_RUN_DIR}/wandb_cache"
export WANDB_DATA_DIR="${CONT_RUN_DIR}/wandb_data"
# Absolutní cesta k checkpointům (Python os.path.join ji použije jako absolutní a ignoruje base_dir)
export CHECKPOINT_SAVE_FOLDER="${CONT_RUN_DIR}/checkpoints"

# --- CACHE SETUP (CRITICAL FOR LUMI/AMD) ---
export HOST_CACHE_ROOT="${PROJECT_DIR}/cache_system_v2"
mkdir -p "${HOST_CACHE_ROOT}/tmp"
mkdir -p "${HOST_CACHE_ROOT}/miopen"
mkdir -p "${HOST_CACHE_ROOT}/triton"
mkdir -p "${HOST_CACHE_ROOT}/torch"
mkdir -p "${HOST_CACHE_ROOT}/hf"
mkdir -p "${HOST_CACHE_ROOT}/mpl"

# Cesty uvnitř kontejneru
export CONT_CACHE_ROOT="${MNT_DIR_CONTAINER}/cache_system_v2"

# --- Training Configuration ---
export EPOCHS_FULL=5
export BATCH_SIZE=47
export BASE_DIR="/mnt/data/datasets/"
# export DATASETS_PREFIX="dataset_split_"
export DATASETS_PREFIX="dataset_homology_split_rev_fixed_"
export PROJECT_NAME="protein-mutation-prediction-protbert-composer"
export MODEL_NAME="Rostlab/prot_bert_bfd"
export SEQ_WINDOW_SIZE=255
export FREEZED_LAYERS=3

export CPU_BIND_MASKS="0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000"

# Tell RCCL to use Slingshot interfaces and GPU RDMA
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB
# ------------------------------

export CONTAINER=/flash/project_465002740/containers/rocm-esm-wandb-composer-rocm7.2.sif

echo "Spouštím trénink na uzlu: $(hostname)"
echo "Dostupná GPU zařízení: $CUDA_VISIBLE_DEVICES"


printf "Starting training script \n \n:"
srun --cpu-bind=v,mask_cpu:$CPU_BIND_MASKS singularity exec --rocm -B ${PROJECT_DIR}:${MNT_DIR_CONTAINER} \
    --env WANDB_DIR=${WANDB_DIR} \
    --env WANDB_CACHE_DIR=${WANDB_CACHE_DIR} \
    --env WANDB_DATA_DIR=${WANDB_DATA_DIR} \
    --env HF_HOME=${CONT_CACHE_ROOT}/hf \
    --env TORCH_HOME=${CONT_CACHE_ROOT}/torch \
    --env XDG_CACHE_HOME=${CONT_CACHE_ROOT}/xdg_cache \
    --env TRITON_HOME=${CONT_CACHE_ROOT}/triton \
    --env TRITON_CACHE_DIR=${CONT_CACHE_ROOT}/triton \
    --env MIOPEN_USER_DB_PATH=${CONT_CACHE_ROOT}/miopen \
    --env MIOPEN_CUSTOM_CACHE_DIR=${CONT_CACHE_ROOT}/miopen \
    --env MPLCONFIGDIR=${CONT_CACHE_ROOT}/mpl \
    --env TMPDIR=${CONT_CACHE_ROOT}/tmp \
    --env TMP=${CONT_CACHE_ROOT}/tmp \
    --env TEMP=${CONT_CACHE_ROOT}/tmp \
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


#!/bin/bash
#SBATCH --job-name=esm_composer_train
#SBATCH --account=project_465002373
#SBATCH --partition=standard-g
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NPROCS
export LOCAL_WORLD_SIZE=$SLURM_GPUS_PER_NODE


# Nastavení cest na hostovi
export PROJECT_DIR=/flash/project_465002373/protbert
export MNT_DIR_CONTAINER=/mnt/data

# --- Time-based Directory Setup ---
export START_TIME=$(date +"%Y-%m-%d_%H-%M-%S")
export HOST_RUN_DIR="${PROJECT_DIR}/runs-esm/${START_TIME}"
mkdir -p "${HOST_RUN_DIR}"

# Cesta uvnitř KONTEJNERU
export CONT_RUN_DIR="${MNT_DIR_CONTAINER}/runs-esm/${START_TIME}"

# --- CACHE SETUP (CRITICAL FIX FOR LLVM ERROR) ---
# Vytvoříme složky na hostovi (scratch filesystém)
export HOST_CACHE_ROOT="${PROJECT_DIR}/cache_system_v2" # Změnil jsem název pro čistý start
mkdir -p "${HOST_CACHE_ROOT}/tmp"
mkdir -p "${HOST_CACHE_ROOT}/miopen"  # <--- CRITICAL PRO AMD
mkdir -p "${HOST_CACHE_ROOT}/triton"
mkdir -p "${HOST_CACHE_ROOT}/torch"
mkdir -p "${HOST_CACHE_ROOT}/hf"
mkdir -p "${HOST_CACHE_ROOT}/mpl"     # Matplotlib cache

# Cesty uvnitř kontejneru (musí odpovídat bind mountu)
export CONT_CACHE_ROOT="${MNT_DIR_CONTAINER}/cache_system_v2"

# --- Training Configuration ---
export EPOCHS_FULL=5
export BATCH_SIZE=2
export BASE_DIR="/mnt/data/datasets/"
export DATASETS_PREFIX="dataset_homology_split_"
export PROJECT_NAME="protein-mutation-prediction-esm2-composer"
export MODEL_NAME="facebook/esm2_t36_3B_UR50D"
export SEQ_WINDOW_SIZE=510
export FREEZED_LAYERS=3
export CHECKPOINT_SAVE_FOLDER="${CONT_RUN_DIR}/checkpoints"

# WandB paths
export WANDB_DIR="${CONT_RUN_DIR}/wandb"
export WANDB_CACHE_DIR="${CONT_CACHE_ROOT}/wandb_cache"
export WANDB_DATA_DIR="${CONT_CACHE_ROOT}/wandb_data"
mkdir -p "${HOST_RUN_DIR}/wandb" "${HOST_CACHE_ROOT}/wandb_cache" "${HOST_CACHE_ROOT}/wandb_data"

export CPU_BIND_MASKS="0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000"
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB

export CONTAINER=${PROJECT_DIR}/rocm-protbert-wand-composer-rocm7.1.1.sif

echo "Spouštím trénink..."

# --- SPUŠTĚNÍ S MIOPEN FIXEM ---
# Důležité: Přidány proměnné MIOPEN_USER_DB_PATH a TRITON_HOME
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
    composer ${MNT_DIR_CONTAINER}/train_composer_esm.py \
    --batch_size ${BATCH_SIZE} \
    --base_dir="${BASE_DIR}" \
    --datasets_prefix="${DATASETS_PREFIX}" \
    --project_name="${PROJECT_NAME}"  \
    --model_name="${MODEL_NAME}" \
    --seq_window_size ${SEQ_WINDOW_SIZE} \
    --freezed_layers ${FREEZED_LAYERS} \
    --epochs ${EPOCHS_FULL} \
    --save_folder "${CHECKPOINT_SAVE_FOLDER}"
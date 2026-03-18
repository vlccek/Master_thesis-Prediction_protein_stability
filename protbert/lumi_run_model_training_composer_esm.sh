#!/bin/bash
#SBATCH --job-name=esm_composer
#SBATCH --account=project_465002740
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=13:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# --- Network & Distributed Setup   ---
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export GPUS_PER_NODE=8
export NNODES=$SLURM_NNODES
export WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

echo "Master Addr: $MASTER_ADDR"
echo "World Size: $WORLD_SIZE ($NNODES nodes x $GPUS_PER_NODE gpus)"

# --- Paths & Project Setup ---
export PROJECT_DIR=/flash/project_465002740/protbert
export HTML_TEMPLATES_DIR=/flash/project_465002740/html_templates
export MNT_DIR_CONTAINER=/mnt/data

# --- Time-based Directory Setup ---
export START_TIME=$(date +"%Y-%m-%d_%H-%M-%S")
export HOST_RUN_DIR="${PROJECT_DIR}/runs-esm/${START_TIME}"
mkdir -p "${HOST_RUN_DIR}"

# Cesta uvnitř KONTEJNERU
export CONT_RUN_DIR="${MNT_DIR_CONTAINER}/runs-esm/esm511${START_TIME}"

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
export BATCH_SIZE=32
export BASE_DIR="/mnt/data/datasets/"
export DATASETS_PREFIX="dataset_homology_split_"
export PROJECT_NAME="ESM-Siamese"
export MODEL_NAME="facebook/esm2_t33_650M_UR50D"
export SEQ_WINDOW_SIZE=500
export CHECKPOINT_SAVE_FOLDER="${CONT_RUN_DIR}/checkpoints"

# WandB paths
export WANDB_DIR="${CONT_RUN_DIR}/wandb"
export WANDB_CACHE_DIR="${CONT_CACHE_ROOT}/wandb_cache"
export WANDB_DATA_DIR="${CONT_CACHE_ROOT}/wandb_data"
mkdir -p "${HOST_RUN_DIR}/wandb" "${HOST_CACHE_ROOT}/wandb_cache" "${HOST_CACHE_ROOT}/wandb_data"

# --- LUMI Specific Hardware Settings ---
# Upravená maska pro 1 task na uzel (pokrývá všechna jádra, Composer si to přebere)
# Pokud by to dělalo problémy, lze masku odstranit, ale na LUMI je doporučeno vázat.
# Zde používáme masku, která dovolí procesu vidět vše, protože Composer si thready managuje sám.
export CPU_BIND="mask_cpu:0xffffffffffffff00"
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_DEBUG=INFO # Pro debugování, pokud se to zasekne

export CONTAINER=/flash/project_465002740/containers/rocm-esm-wandb-composer-rocm7.2.sif

# --- Command Construction ---

# Příkaz pro spuštění uvnitř kontejneru
# Používáme proměnné předané přes srun/bash
export COMPOSER_ARGS="--world_size $WORLD_SIZE \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT"

export SCRIPT_ARGS="--batch_size ${BATCH_SIZE} \
    --base_dir ${BASE_DIR} \
    --datasets_prefix ${DATASETS_PREFIX} \
    --project_name ${PROJECT_NAME} \
    --model_name ${MODEL_NAME} \
    --seq_window_size ${SEQ_WINDOW_SIZE} \
    --epochs ${EPOCHS_FULL} \
    --save_folder ${CHECKPOINT_SAVE_FOLDER} \
    --lr 1e-4 "

echo "Spouštím multi-node trénink na $NNODES uzlech..."

# --- SPUŠTĚNÍ ---
# 1. srun spustí 1 proces na každém uzlu (--ntasks-per-node=1)
# 2. singularity spustí kontejner
# 3. bash uvnitř kontejneru zjistí NODE_RANK ze SLURM_PROCID a spustí composer
# 4. composer spustí 8 workerů (pro každou GPU)

srun \
    --ntasks-per-node=1 \
    singularity exec --rocm -B ${PROJECT_DIR}:${MNT_DIR_CONTAINER} -B ${HTML_TEMPLATES_DIR}:/mnt/html_templates \
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
    --env NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \
    --env NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL} \
    $CONTAINER \
    bash -c "export NODE_RANK=\$SLURM_PROCID && \
             echo \"Node Rank: \$NODE_RANK / World Size: $WORLD_SIZE\" && \
             composer $COMPOSER_ARGS --node_rank \$NODE_RANK \
             ${MNT_DIR_CONTAINER}/train_composer_esm.py $SCRIPT_ARGS"
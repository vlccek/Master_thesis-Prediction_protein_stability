#!/bin/bash

# Brno universioty of technology: VUT FIT / BUT FIT
# Master thesis
# Predikce vlivu mutací na stabilitu proteinů / Prediction of the Effect of Mutations on Protein Stability
# 
# author: Jakub Vlk
# date: 2026-03-03

#SBATCH --job-name=esm_composer_multinode
#SBATCH --account=project_465002740
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --time=72:00:00
#SBATCH --mem-per-gpu=60G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Nastavení cest
CONTAINER="../containers/rocm-esm-wandb-composer-rocm7.2.sif"
INPUT_FILE="deduplicated_dataset.parquet"
OUTPUT_FILE="final_embeddings.parquet"
MODEL_NAME="facebook/esm2_t48_15B_UR50D"

# --- CACHE SETUP ---
export HOST_CACHE_ROOT="$(pwd)/cache_inference"
mkdir -p "${HOST_CACHE_ROOT}/tmp"
mkdir -p "${HOST_CACHE_ROOT}/miopen"
mkdir -p "${HOST_CACHE_ROOT}/triton"
mkdir -p "${HOST_CACHE_ROOT}/torch"
mkdir -p "${HOST_CACHE_ROOT}/hf"
mkdir -p "${HOST_CACHE_ROOT}/mpl"
mkdir -p "${HOST_CACHE_ROOT}/xdg_cache"

# Cesty uvnitř kontejneru
export CONT_CACHE_ROOT="/mnt/cache_inference"

BATCH_SIZE=8

echo "Spouštím ESM2 15B inference na AMD GPU..."

# --rocm zajistí přístup k AMD ovladačům
# -B .:/mnt namapuje aktuální složku do /mnt v kontejneru
singularity exec --rocm \
    -B .:/mnt \
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
    "$CONTAINER" \
    python3 /mnt/esm2_inference_v2.py \
    --input "/mnt/$INPUT_FILE" \
    --output "/mnt/$OUTPUT_FILE" \
    --model "$MODEL_NAME" \
    --batch_size $BATCH_SIZE \
    --tmp_dir "/mnt/tmp_chunks"

echo "Inference dokončena."

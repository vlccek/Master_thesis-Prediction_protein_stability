#!/bin/bash
#SBATCH --job-name=bench_inference
#SBATCH --account=project_465002740
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=inference-logs/bench_%j.out
#SBATCH --error=inference-logs/bench_%j.err

# --- 1. SETTINGS ---
export TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
echo "Benchmark Run Start Time: $TIMESTAMP"

# Paths on HOST
# Use SLURM_SUBMIT_DIR if available (set by sbatch), otherwise fallback to current directory
WORKING_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

# If the working directory is the benchmarks folder, the project root is one level up
if [[ "$WORKING_DIR" == */benchmarks ]]; then
    export PROJECT_ROOT="$(dirname "$WORKING_DIR")"
else
    export PROJECT_ROOT="$WORKING_DIR"
fi

export BENCHMARKS_DIR="${PROJECT_ROOT}/benchmarks"
export HOST_RUN_DIR="${BENCHMARKS_DIR}/runs/benchmark_${TIMESTAMP}"

# Container and Mounts
export MNT_ROOT="/mnt/data"
export CONTAINER="${PROJECT_ROOT}/containers/rocm-esm-wandb-composer-rocm7.2.sif"
export PY_SCRIPT="${MNT_ROOT}/benchmarks/run_inference.py"

# --- CACHE SETUP (CRITICAL FOR LUMI/AMD) ---
export HOST_CACHE_ROOT="${PROJECT_ROOT}/protbert/cache_system_v2"
export CONT_CACHE_ROOT="${MNT_ROOT}/protbert/cache_system_v2"

mkdir -p "${HOST_CACHE_ROOT}"/{tmp,miopen,triton,xdg_cache,hf}

# Use SINGULARITYENV_ prefix to propagate these to the container
export SINGULARITYENV_XDG_CACHE_HOME="${CONT_CACHE_ROOT}/xdg_cache"
export SINGULARITYENV_TMPDIR="${CONT_CACHE_ROOT}/tmp"

# Hugging Face caches (critical for AutoTokenizer)
export SINGULARITYENV_HF_HOME="${CONT_CACHE_ROOT}/hf"
export SINGULARITYENV_TRANSFORMERS_CACHE="${CONT_CACHE_ROOT}/hf"

# Critical AMD/Triton paths that sometimes ignore XDG
export SINGULARITYENV_TRITON_CACHE_DIR="${CONT_CACHE_ROOT}/triton"
export SINGULARITYENV_MIOPEN_USER_DB_PATH="${CONT_CACHE_ROOT}/miopen"

# Python specific paths
export SINGULARITYENV_PYTHONPATH="${MNT_ROOT}"

mkdir -p "${HOST_RUN_DIR}"
echo "Output directory: ${HOST_RUN_DIR}"

# --- 2. CONFIGURATION (CHANGE CHECKPOINTS HERE) ---
# ProtBERT checkpoint
export PROTBERT_CHECKPOINT="${PROJECT_ROOT}/protbert/runs/2026-03-28_22-36-45/checkpoints/model_epoch_3.pt"

# ESM checkpoint
export ESM_CHECKPOINT="${PROJECT_ROOT}/protbert/runs-esm/esm5112026-03-31_02-14-07/checkpoints/best_frequent_epoch_1.pt"

# ESM MeanPooling checkpoint
export ESM_MEANPOOL_CHECKPOINT="${PROJECT_ROOT}/protbert/runs-esm/esm5112026-03-27_23-53-49/checkpoints/best_frequent_epoch_0.pt"
# export ESM_MEANPOOL_CHECKPOINT="${PROJECT_ROOT}/protbert/runs-esm-meanpool/2026-03-31_20-39-35//checkpoints/best_frequent_epoch_0.pt"

#
export BATCH_SIZE=64
export PYTHONPATH="${MNT_ROOT}"

# --- 3. RUNNING INFERENCE ---

run_bench() {
    local dataset=$1   # s350 or ponsol
    local model=$2     # protbert or esm
    local host_checkpoint=$3

    echo "--------------------------------------------------------"
    echo "Running: Dataset=$dataset, Model=$model"
    echo "Host Checkpoint: $host_checkpoint"
    
    # Translate host checkpoint path to container path
    # Removes the PROJECT_ROOT prefix and replaces it with MNT_ROOT
    # Robust removal of PROJECT_ROOT regardless of trailing slashes
    local root_trimmed="${PROJECT_ROOT%/}"
    local rel_path="${host_checkpoint#$root_trimmed}"
    # Ensure rel_path starts with a slash
    [[ "$rel_path" != /* ]] && rel_path="/$rel_path"
    local container_checkpoint="${MNT_ROOT}${rel_path}"
    
    echo "Container Checkpoint: $container_checkpoint"
    
    local input_path="${MNT_ROOT}/benchmarks/datasets/${dataset}_prepared.parquet"
    if [ "$dataset" == "s350" ]; then
        input_path="${MNT_ROOT}/benchmarks/datasets/s350_dataset.parquet"
    fi
    
    local output_path="${MNT_ROOT}/benchmarks/runs/benchmark_${TIMESTAMP}/${dataset}_${model}_results.csv"
    
    srun singularity exec --rocm \
        -B ${PROJECT_ROOT}:${MNT_ROOT} \
        $CONTAINER \
        python3 ${PY_SCRIPT} \
        --input "${input_path}" \
        --checkpoint "${container_checkpoint}" \
        --output "${output_path}" \
        --model_type "${model}" \
        --batch_size ${BATCH_SIZE}
}

# Run ProtBERT on all datasets
run_bench "s350" "protbert" "${PROTBERT_CHECKPOINT}"
run_bench "ponsol" "protbert" "${PROTBERT_CHECKPOINT}"
run_bench "benchstab" "protbert" "${PROTBERT_CHECKPOINT}"

# Run ESM on all datasets
run_bench "s350" "esm" "${ESM_CHECKPOINT}"
run_bench "ponsol" "esm" "${ESM_CHECKPOINT}"
run_bench "benchstab" "esm" "${ESM_CHECKPOINT}"

# Run ESM MeanPooling on all datasets
run_bench "s350" "esm_meanpool" "${ESM_MEANPOOL_CHECKPOINT}"
run_bench "ponsol" "esm_meanpool" "${ESM_MEANPOOL_CHECKPOINT}"
run_bench "benchstab" "esm_meanpool" "${ESM_MEANPOOL_CHECKPOINT}"

echo "========================================================"
echo "Done. All benchmarks completed."
echo "Results are stored in: ${HOST_RUN_DIR}"
echo "========================================================"

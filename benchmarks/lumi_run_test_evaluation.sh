#!/bin/bash
#SBATCH --job-name=test_eval_composer
#SBATCH --account=project_465002740
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=04:30:00
#SBATCH --output=eval-inference-logs/test_eval_comp_%j.out
#SBATCH --error=eval-inference-logs/test_eval_comp_%j.err

# --- 1. SETTINGS ---
export TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "Test Evaluation (Composer) Run Start Time: $TIMESTAMP"

# Paths on HOST
WORKING_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ "$WORKING_DIR" == */benchmarks ]]; then
    export PROJECT_ROOT="$(dirname "$WORKING_DIR")"
else
    export PROJECT_ROOT="$WORKING_DIR"
fi

export MNT_ROOT="/mnt/data"
export CONTAINER="${PROJECT_ROOT}/containers/rocm-esm-wandb-composer-rocm7.2.sif"

# --- CACHE SETUP ---
export HOST_CACHE_ROOT="${PROJECT_ROOT}/protbert/cache_system_v2"
export CONT_CACHE_ROOT="${MNT_ROOT}/protbert/cache_system_v2"
mkdir -p "${HOST_CACHE_ROOT}"/{tmp,miopen,triton,xdg_cache,hf,torch,mpl}

# --- CONFIGURATION ---
export PROTBERT_CHECKPOINT="${PROJECT_ROOT}/protbert/runs/2026-03-28_22-36-45/checkpoints/model_epoch_3.pt"
export ESM_CHECKPOINT="${PROJECT_ROOT}/protbert/runs-esm/esm5112026-03-31_02-14-07/checkpoints/best_frequent_epoch_1.pt"
export ESM_MEANPOOL_CHECKPOINT="${PROJECT_ROOT}/protbert/runs-esm/esm5112026-03-27_23-53-49/checkpoints/best_frequent_epoch_0.pt"

TEST_FILE="${TEST_FILE:-${MNT_ROOT}/protbert/datasets/dataset_homology_split_rev_fixed_test.csv}"
BATCH_SIZE="${BATCH_SIZE:-47}"
BASE_OUTPUT_DIR="${MNT_ROOT}/benchmarks/runs/test_eval_comp_${TIMESTAMP}"

# SLURM / MPI / Composer distribution settings
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12345

run_eval() {
    local model_type=$1
    local host_checkpoint=$2
    
    echo "--------------------------------------------------------"
    echo "Evaluating Model: $model_type on 8 GPUs"
    
    local root_trimmed="${PROJECT_ROOT%/}"
    local rel_path="${host_checkpoint#$root_trimmed}"
    [[ "$rel_path" != /* ]] && rel_path="/$rel_path"
    local container_checkpoint="${MNT_ROOT}${rel_path}"
    
    local output_dir="${BASE_OUTPUT_DIR}/${model_type}"

    # Use srun to launch 8 processes (one per GPU)
    srun singularity exec --rocm -B ${PROJECT_ROOT}:${MNT_ROOT} \
        --env HF_HOME=${CONT_CACHE_ROOT}/hf \
        --env TRANSFORMERS_CACHE=${CONT_CACHE_ROOT}/hf \
        --env TORCH_HOME=${CONT_CACHE_ROOT}/torch \
        --env XDG_CACHE_HOME=${CONT_CACHE_ROOT}/xdg_cache \
        --env TRITON_HOME=${CONT_CACHE_ROOT}/triton \
        --env TRITON_CACHE_DIR=${CONT_CACHE_ROOT}/triton \
        --env MIOPEN_USER_DB_PATH=${CONT_CACHE_ROOT}/miopen \
        --env MIOPEN_CUSTOM_CACHE_DIR=${CONT_CACHE_ROOT}/miopen \
        --env MPLCONFIGDIR=${CONT_CACHE_ROOT}/mpl \
        --env TMPDIR=${CONT_CACHE_ROOT}/tmp \
        --env PYTHONPATH="${MNT_ROOT}" \
        $CONTAINER \
        composer ${MNT_ROOT}/benchmarks/run_test_evaluation_composer.py \
        --test_file "${TEST_FILE}" \
        --checkpoint "${container_checkpoint}" \
        --output_dir "${output_dir}" \
        --model_type "${model_type}" \
        --batch_size "${BATCH_SIZE}"
}

run_eval "protbert" "${PROTBERT_CHECKPOINT}"
# run_eval "esm" "${ESM_CHECKPOINT}"
# run_eval "esm_meanpool" "${ESM_MEANPOOL_CHECKPOINT}"

echo "========================================================"
echo "Evaluation complete. Results in ${BASE_OUTPUT_DIR}"
echo "========================================================"

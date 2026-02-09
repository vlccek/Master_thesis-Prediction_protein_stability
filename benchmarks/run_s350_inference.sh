#!/bin/bash
#SBATCH --job-name=protbert_inference
#SBATCH --account=project_465002373
#SBATCH --partition=standard-g
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=logs/inference_%j.out
#SBATCH --error=logs/inference_%j.err

# --- 1. Nastavení proměnných a času ---
export TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
echo "Inference Run Start Time: $TIMESTAMP"

# Hlavní cesty na HOSTITELE (fyzický stroj)
export PROJECT_ROOT="/flash/project_465002373"
export BENCHMARKS_DIR="${PROJECT_ROOT}/benchmarks"
# Tady vytvoříme tu složku s datem a časem
export HOST_OUTPUT_DIR="${BENCHMARKS_DIR}/runs/s358_${TIMESTAMP}"

# Vytvoření výstupní složky
mkdir -p "${HOST_OUTPUT_DIR}"
echo "Vytvořena výstupní složka: ${HOST_OUTPUT_DIR}"

# --- 2. Konfigurace Kontejneru ---
# Mapujeme celý PROJECT_ROOT do /mnt/data, abychom viděli jak na model, tak na data
export MNT_ROOT="/mnt/data"
export CONTAINER="${PROJECT_ROOT}/protbert/rocm-protbert-wand-composer-rocm7.1.1.sif"

# Cesta k python scriptu (předpokládám, že je ve složce protbert nebo benchmarks, upravte dle reality)
# Pokud je inference.py ve složce benchmarks na hostiteli:
PY_SCRIPT="${MNT_ROOT}/benchmarks/run_inference.py"

# Vstupní data
PY_INPUT="${MNT_ROOT}/benchmarks/datasets/s350_dataset.parquet"

# Checkpoint modelu (upravte cestu k modelu, pokud je jinde)
PY_CHECKPOINT="${MNT_ROOT}/protbert/runs/2026-02-02_13-41-02/checkpoints/model_epoch_5.pt"

# Výstupní soubor (CSV) - uložíme ho do té časové složky
# Python script k němu automaticky přihodí i metrics.txt
PY_OUTPUT="${MNT_ROOT}/benchmarks/runs/s500_${TIMESTAMP}/results.csv"

export BATCH_SIZE=64

# --- 4. Spuštění ---
echo "Spouštím inferenci..."
echo "Input: $PY_INPUT"
echo "Output Dir: $PY_OUTPUT"


export PYTHONPATH="${MNT_ROOT}"

# -B ${PROJECT_ROOT}:${MNT_ROOT} zpřístupní vše pod project_465002373
srun singularity exec --rocm \
    -B ${PROJECT_ROOT}:${MNT_ROOT} \
    $CONTAINER \
    python3 ${PY_SCRIPT} \
    --input "${PY_INPUT}" \
    --checkpoint "${PY_CHECKPOINT}" \
    --output "${PY_OUTPUT}" \
    --batch_size ${BATCH_SIZE}

# (Volitelné) Kopírování logu SLURMu do výsledné složky pro archivaci
# cp logs/inference_${SLURM_JOB_ID}.out "${HOST_OUTPUT_DIR}/" 2>/dev/null

echo "========================================================"
echo "Hotovo."
echo "Výsledky (CSV + metriky) jsou uloženy v:"
echo "${HOST_OUTPUT_DIR}"
echo "========================================================"
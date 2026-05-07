#!/bin/bash

# Brno universioty of technology: VUT FIT / BUT FIT
# Master thesis
# Predikce vlivu mutací na stabilitu proteinů / Prediction of the Effect of Mutations on Protein Stability
# 
# author: Jakub Vlk
# date: 2025-12-12

#PBS -N ipro_scan_v6
#PBS -l select=1:ncpus=4:mem=16gb:scratch_local=40gb
#PBS -l walltime=6:00:00
#PBS -J 0-1


# 194
# === CESTY ===
BASE_DIR="$PBS_O_WORKDIR"
INPUT_DIR="$BASE_DIR/input_chunks"
OUTPUT_DIR="$BASE_DIR/results-tests/"

# Cesty k tomu, co jsme stáhli v kroku 2
SETUP_DIR="$BASE_DIR"
IMAGE="$BASE_DIR/interproscan.sif"
DATA_PATH="$BASE_DIR/interproscan-5.76-107.0/data"

mkdir -p "$OUTPUT_DIR"

# Určení souboru podle ID jobu (0000, 0001...)
CHUNK_ID=$(printf "%04d" $PBS_ARRAY_INDEX)
INPUT_FILE="chunk_${CHUNK_ID}.fasta"
OUTPUT_FILE="result_${CHUNK_ID}.tsv"

# Kontrola
if [ ! -f "$INPUT_DIR/$INPUT_FILE" ]; then
    echo "Chyba: Vstup $INPUT_FILE neexistuje!"
    exit 1
fi

# Kopírování na scratch
cp "$INPUT_DIR/$INPUT_FILE" "$SCRATCHDIR/input.fasta"
cd "$SCRATCHDIR"

echo "Spouštím InterProScan 6 na $HOSTNAME..."

# === SPUŠTĚNÍ SINGULARITY ===
# -B: Mapuje složku s daty do kontejneru tam, kde je program čeká (/opt/interproscan/data)
singularity exec -B "$DATA_PATH":/opt/interproscan/data -B $SCRATCHDIR:/tmp "$IMAGE" \
    /opt/interproscan/interproscan.sh \
    -i input.fasta \
    -f tsv \
    -dp \
    -appl Pfam,Gene3D \
    -cpu 4 \
    -T $SCRATCHDIR \
    -o output.tsv


# Uložení výsledků
if [ -f "output.tsv" ]; then
    cp output.tsv "$OUTPUT_DIR/$OUTPUT_FILE"
else
    # Vytvoří prázdný soubor, abychom věděli, že job proběhl (i když nic nenašel)
    touch "$OUTPUT_DIR/$OUTPUT_FILE"
fi

clean_scratch
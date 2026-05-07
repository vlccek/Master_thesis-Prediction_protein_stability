#!/bin/bash

# Brno universioty of technology: VUT FIT / BUT FIT
# Master thesis
# Predikce vlivu mutací na stabilitu proteinů / Prediction of the Effect of Mutations on Protein Stability
# 
# author: Jakub Vlk
# date: 2025-12-12


TARGET_DIR="./interpro_data"
VERSION="5.76-107.0"

mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

echo "1. Stahování InterProScan Core (verze 5.65-97.0)..."
# Odkaz se může časem měnit, zkontroluj na EBI webu, pokud toto selže
wget -nc https://ftp.ebi.ac.uk/pub/software/unix/iprscan/5/5.65-97.0/interproscan-5.65-97.0-64-bit.tar.gz

echo "2. Rozbalování (trvá cca 10-15 min)..."
tar -pxzf interproscan-5.65-97.0-64-bit.tar.gz

echo "3. Stahování Singularity image..."
singularity pull --name interproscan.sif docker://interpro/interproscan:latest

echo "HOTOVO!"
echo "Data rozbalena v: $TARGET_DIR/interproscan-5.65-97.0/data"
echo "Image uložen v: $TARGET_DIR/interproscan.sif"


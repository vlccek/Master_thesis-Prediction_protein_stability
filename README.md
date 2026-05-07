# Master's Thesis: Protein Stability Research

This repository contains tools and datasets for protein stability prediction and analysis, focusing on DDG mutation effects using models like ProtBert and ESM-2. This project is a core component of a Master's Thesis.

**Thesis Text & LaTeX:** [thesis_text/](./thesis_text) (Git Submodule)
**Excel@FIT Poster:** [posters/](./posters) (Exhibited at [Excel@FIT](https://excel.fit.vutbr.cz/))

## Repository Map

### Core Workflows
- **`protbert/`**: Primary training and model definition directory. Includes scripts for ProtBert and ESM-2 using the Composer framework.
- **`clustering/`**: ESM-2 based sequence inference and clustering logic for dataset preparation.
- **`benchmarks/`**: Scripts and datasets for evaluating model performance against standard benchmarks (S350, BenchStab, PonSol). **Also serves as a reference for loading and performing inference with trained checkpoints.**

### Data & Datasets
- **`data/`**: Raw dataset files (CSV, TXT, TSV) including MegaScale, Lehner, and others.
- **`classification_dataset/`**: Scripts for building and splitting datasets for classification tasks.
- **`clustering/datasets/`**: Prepared datasets for benchmarking.

### Infrastructure & Environment
- **`containers/`**: Location for Singularity images (`.sif`) for ROCm/PyTorch environments (not presented in the repository).
- **`singularity_conf/`**: Definition files for building the execution environments.

### Analysis & Visualization
- **`fig-generation/`**: Notebooks for generating figures, heatmaps, and CATH hierarchy visualizations for thesis text.
- **`posters/`**: Conference poster from Excel@FIT.
- **`html_templates/`**: Templates for generating interactive reports.

## Getting Started
Execution environments are managed via Singularity/Apptainer ment to be used on LUMI supercomputer. Note that `.sif` files are not tracked in the repository; they must be built using the definition files provided in `singularity_conf/`.

### Primary Model Checkpoints (unavailable at github)
The following checkpoints are used in the benchmark scripts (`benchmarks/run_all_benchmarks.sh`):
- **ProtBERT:** `protbert/runs/2026-03-28_22-36-45/checkpoints/model_epoch_3.pt`
- **ESM:** `protbert/runs-esm/esm5112026-03-31_02-14-07/checkpoints/best_frequent_epoch_1.pt`
- **ESM MeanPooling:** `protbert/runs-esm/esm5112026-03-27_23-53-49/checkpoints/best_frequent_epoch_0.pt`


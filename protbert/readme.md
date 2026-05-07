# ProtBert & ESM Model Training - Master thesis

This directory contains the core logic for training and evaluating protein stability prediction models (ProtBert, ESM-2)
using the Composer framework. The work is part of the master's thesis by Jakub Vlk, carried out in the 2025/26
academic year.

## Core Components

### Model definitions

- **Model.py**: Base implementation of the ProtBert model.
- **ModelComposerESM.py**: ESM-2 integration with Composer.
- **ModelComposerESM_meanpooling.py**: ESM-2 with mean-pooling architecture.

### Running training on LUMI

- **lumi_run_model_trainin\*.sh**: Batch job scripts for training different models.

### Training scripts

- **train_composer.py**: Main entry point for ProtBert training.
- **train_composer_esm.py**: Training script for ESM-2 models.
- **train_composer_esm_meanpooling.py**: Training script for ESM-2 with mean pooling.

### Execution scripts (HPC)

- **lumi_run_model_training_composer.sh**: Job script for LUMI (ProtBert).
- **lumi_run_model_training_composer_esm.sh**: Job script for LUMI (ESM).
- **run_model_training.sh**: Legacy/generic training script for MetaCentrum.

## Dataset preparation workflow

Detailed documentation on data processing can be found in [DATASET_WORKFLOW.md](DATASET_WORKFLOW.md).

1. **Lehner dataset:** `lehner_dataset_preparation.ipynb`
2. **Megascale dataset:** `megascale_dataset_preparation.ipynb` & `megascale_dataset_normalization.ipynb`
3. **Merging & CATH annotation:** `merging_datasets.ipynb`
4. **Final split generation:** `dataset_prepare_*.ipynb` (creates train/val/test splits used for training)

## Environment setup

### Containerized execution

Use the Singularity definition files in the `singularity_conf/` directory to build containers for the LUMI environment (
AMD GPUs: MI250).

## Authorship

Files marked with the authorship of the thesis author (Jakub Vlk) belong to him. If a file does not include an
authorship comment, please refer to the nearest README.md for the authorship information.
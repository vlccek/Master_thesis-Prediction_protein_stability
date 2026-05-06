# Workflow for Dataset Preparation and Normalization

**Updated:** 2026-03-21

This document describes the complete data preparation workflow for the two primary datasets used in this project: **Lehner** and **Megascale**.

---

## Dataset Overview

| Dataset | Source | Target Metric |
|---------|--------|---------------|
| Lehner | [Lehner et al.](https://www.nature.com/articles/nchembio.1174) | `normalized_fitness` |
| Megascale | [Megascale project](https://www.science.org/doi/10.1126/science.ade6353) | `ddG` (stability change) |

---

## 1. Lehner Dataset Preparation

### Input
- File: `../data/lehner_dataset.txt` (tab-separated)

### Workflow Steps (`lehner_dataset_preparation.ipynb`)

1. **Load raw data**
   ```python
   df = pl.read_csv("../data/lehner_dataset.txt", separator="\t", null_values="NA")
   ```
   - Initial size: ~602,882 mutations
   - Key columns: `domain_ID`, `uniprot_ID`, `aa_seq`, `wt_aa`, `position`, `mut_aa`, `fitness`, `normalized_fitness`, `fitness_sigma`, `quality_rank`

2. **Fetch parent sequences from UniProt**
   - Query UniProt API for original protein sequences
   - Save mapping to `uniprot_id_to_sequence_mapping.json`
   - Fall back to manual mapping for failed IDs

3. **Create reverse mutations**
   - For each mutation, create a reverse counterpart
   - Swap `wt_aa` and `mut_aa`
   - Negate `normalized_fitness` value
   - Add `reverse` boolean column
   - Concatenate original + reverse datasets
   - **Result:** ~1,205,764 rows (doubled due to reverse mutations)

4. **Compute sigmoid normalization**
   - Formula (piecewise sigmoid/exponential):
     - For non-negative values: `f(x) = A_pos * (2 / (1 + exp(-k_pos * x)) - 1)`
     - For negative values: `f(x) = -A_neg * (2 / (1 + exp(-k_neg * (-x))) - 1)`
   - Parameters used: `k_neg = 0.871`, `k_pos = 0.871`, `A_neg = 1.0`, `A_pos = 1.0`
   - Target saturation: 0.99 at extremes
   - **Output column:** `normalized_fitness_sigmoid` in range (-1, 1)

5. **Generate full sequences**
   - Reconstruct full original and mutated sequences using `original_sequence`, `position`, and mutation info
   - Format: `mut_type` like "Q2*", "A2C", etc.
   - Filter out null sequences
   - **Final size:** ~1,116,004 mutations

6. **Output files**
   - `datasets/lehner_dataset_with_sequences_normalized.csv`

### Normalization Details

**Why piecewise sigmoid?**
- Natural fitness/stability values are bounded (measurements have limits)
- Sigmoid compresses extreme values toward saturation (±1)
- Symmetric treatment of positive/negative effects
- Preserves relative ordering while normalizing dynamic range

---

## 2. Megascale Dataset Preparation

### Input
- File: `../data/megascale.csv`

### Workflow Steps (`megascale_dataset_preparation.ipynb`)

1. **Load and clean raw data**
   ```python
   df = pl.read_csv("../data/megascale.csv", null_values=["NA"])
   ```
   - Initial size: ~776,298 measurements

2. **Extract mutation information**
   - Parse `mut_type` (e.g., "D13Q", "I32P:L40I")
   - Extract wild-type residue, position, and mutated residue
   - Convert `deltaG` and `ddG_ML` columns to numeric

3. **Reconstruct original sequences**
   - For each mutation, revert the mutation to get original sequence
   - Handle multi-mutation entries by reverting all mutations
   - **Result:** Each mutation has `original_seq_full` and `mutated_seq_full`

4. **Calculate ddG values**
   - Join with WT measurements (same sequence, no mutation)
   - Compute: `ddG = deltaG_mutant - deltaG_wildtype`
   - Group by sequence and average replicate measurements
   - **Final size (after filtering):** ~416,914 unique mutations

### Normalization (`megascale_dataset_normalization.ipynb`)

1. **Apply piecewise sigmoid** (same formula as Lehner)
   - Parameters: `k_neg = 0.230`, `k_pos = 0.576`, `A_neg = 1.0`, `A_pos = 1.0`
   - Different parameters due to different data distribution
   - Target column: `normalized_stability`

2. **Create reverse mutations**
   - Swap `original_seq_full` <-> `mutated_seq_full`
   - Negate `normalized_stability`
   - Add `reverse` boolean column
   - Concatenate original + reverse datasets
   - **Final size:** ~833,828 rows

3. **Output file**
   - `datasets/megascale_dataset_with_ddg_normalized.csv`

---

## 3. Dataset Merging

### Workflow (`merging_datasets.ipynb`)

1. **Load normalized datasets**
   - Lehner: `lehner_dataset_with_sequences_normalized.csv`
   - Megascale: `megascale_dataset_with_ddg_normalized.csv`

2. **Standardize columns**
   - Rename target columns to `target`
   - Add `data_source` column ("lehner" or "megascale")
   - Select common columns:
     - `original_seq_full`
     - `mutated_seq_full`
     - `mut_type`
     - `target`
     - `reverse`
     - `data_source`

3. **Concatenate datasets**
   - **Combined size:** ~1,949,832 rows
   - Output: `datasets/dataset_merged.parquet`

4. **Add CATH annotations** (optional)
   - Join with CATH domain classification
   - Extract hierarchical levels: class, architecture, topology, homology
   - Output: `datasets/dataset_merged_with_families.parquet`


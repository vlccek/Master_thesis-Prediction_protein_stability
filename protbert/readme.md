- emb.py generation of embeddings
- data_playground.ipynb playground for data exploration, converting them to pkl file.


#  255W training

```
rsync ~/projekty/dp/protbert/ xvlkja07@nympha.meta.zcu.cz:~/dp/protbert/ -rPz  --exclude-from=rsyncignore.txt `
```

- data perations are in dataset_prepare_255w.ipynb
- copy the `dataset_255w_*` to the metacetrum servers
- prepare the enviroment to the env `env.tar.gz` and the `train_dataset.tar.gz` with dataset and python files (`*.py`)
  - by `tar --use-compress-program="pigz -k " -cf training.tar.gz *.py dataset_255w_*.csv`
  - and by `tar --use-compress-program="pigz -k " -cf env.tar.gz env` NOTE: takes a while
- run `qsub run_model_training.sh` dont forget to set parametrs in the `run_model_training.sh` file

# Running singularity container

- prepare the singularity image `singularity build --fakeroot protbert.sif Singularity`
- run the singularity container `singularity shell --bind /path/to/bind:/path

# copy to lumi scratch

```bash
rsync ~/dp/protbert/ /scratch/project_465002373/protbert/ -rPz
```


# Workflow for dataset preparation

**Detailed documentation:** See [DATASET_WORKFLOW.md](DATASET_WORKFLOW.md) for comprehensive information about:
- Data sources and normalization procedures
- Piecewise sigmoid normalization parameters (separate for positive/negative values)
- Complete workflow for both Lehner and Megascale datasets
- File descriptions and statistics

## Overview

1) **Lehner dataset:** `lehner_dataset_preparation.ipynb`
2) **Megascale dataset:** `megascale_dataset_preparation.ipynb` + `megascale_dataset_normalization.ipynb`
3) **Merging datasets:** `merging_datasets.ipynb` (also adds CATH homology annotations)
4) **255W preparation:** `dataset_prepare_255w.ipynb` (creates train/validation/test splits)

## Normalization

Both datasets use **piecewise sigmoid normalization** with separate parameters for positive and negative values:

- **Lehner:** `k_neg = 0.871`, `k_pos = 0.871`
- **Megascale:** `k_neg = 0.230`, `k_pos = 0.576`

See `DATASET_WORKFLOW.md` for formulas and detailed explanation.
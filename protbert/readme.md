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
1) for lehner dataset use `dataset_prepare_lehner.ipynb`
2) for megasccale dataset use `dataset_prepare_megascale.ipynb` and `dataset_megascale_filtering.ipynb`
3) For merging datasets use `dataset_merge_datasets.ipynb` there is also adding the homology anotattions
4) for peraparation of 255w (two columns with 255AA that are used for traing model with context len of 512) dataset use `dataset_prepare_255w.ipynb`
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


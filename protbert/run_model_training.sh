#!/bin/bash

# Brno universioty of technology: VUT FIT / BUT FIT
# Master thesis
# Predikce vlivu mutací na stabilitu proteinů / Prediction of the Effect of Mutations on Protein Stability
# 
# author: Jakub Vlk
# date: 2025-09-21

#PBS -q default@pbs-m1.metacentrum.cz
#PBS -l walltime=120:00:00
#PBS -l select=1:ncpus=4:ngpus=1:mem=44gb:gpu_mem=60gb:scratch_ssd=40gb
#PBS -N protbert_train


module add mambaforge


HOMEDIR=/storage/plzen1/home/xvlkja07

ENV_TAR=env.tar.gz
DATASET_TAR=training.tar.gz

rsync -avzP ${HOMEDIR}/dp/protbert/${ENV_TAR} ${SCRATCHDIR}
rsync -avzP ${HOMEDIR}/dp/protbert/${DATASET_TAR} ${SCRATCHDIR}


cd ${SCRATCHDIR}


pigz -dc ${ENV_TAR} | tar xf -
pigz -dc ${DATASET_TAR} | tar xf -


mamba activate env/


cd ${SCRATCHDIR}

python train_single.py --epochs 5 --step_validation 10000 --smart_batch 1


echo "Training finished, copying the model back to home directory"
rsync -avzP ${SCRATCHDIR}/best_single_model.pth ${HOMEDIR}/dp/protbert/models/best_single_model.pth

echo "All done"

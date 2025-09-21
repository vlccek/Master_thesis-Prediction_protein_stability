#!/bin/bash
#PBS -q default@pbs-m1.metacentrum.cz
#PBS -l walltime=1:00:0
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb:gpu_mem=24gb:scratch_local=40gb
#PBS -N protbert_train


module add mambaforge


HOMEDIR=/storage/plzen1/home/xvlkja07

rsync -avzP ${HOMEDIR}/dp/protbert/env.tar.gz ${SCRATCHDIR}
rsync -avzP ${HOMEDIR}/dp/protbert/train_dataset.tar.gz ${SCRATCHDIR}


cd ${SCRATCHDIR}


pigz -dc env.tar.gz | tar xf -
pigz -dc train_dataset.tar.gz | tar xf -


mamba activate env/


cd ${SCRATCHDIR}

lrs=( 0.000001  0.000005  0.0000001 0.0000005 0.00000005)

# Fixed params
limit=10000
epochs=3

# Loop over learning rates
for lr in "${lrs[@]}"; do
    echo "Running training with lr=$lr"
    python train_single.py \
        --limit ${limit} \
        --lr ${lr} \
        --project_name "protein-mutation-prediction-protbert-lr-optimization"
done



rsync -avzP ${SCRATCHDIR}/wandb/ ${HOMEDIR}/dp/protbert/wandb/
rsync -avzP ${SCRATCHDIR}/best_single_model.pth ${HOMEDIR}/dp/protbert/models/best_single_model.pth

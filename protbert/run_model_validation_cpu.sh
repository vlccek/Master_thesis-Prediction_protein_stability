#!/bin/bash
#PBS -q default@pbs-m1.metacentrum.cz
#PBS -l walltime=00:10:00
#PBS -l select=1:ncpus=32:mem=128gb:scratch_ssd=40gb
#PBS -N protbert_train


module add mambaforge


HOMEDIR=/storage/plzen1/home/xvlkja07

rsync -avzP ${HOMEDIR}/dp/protbert/env.tar.gz ${SCRATCHDIR}
rsync -avzP ${HOMEDIR}/dp/protbert/validation.tar.gz ${SCRATCHDIR}


cd ${SCRATCHDIR}


pigz -dc env.tar.gz | tar xf -
pigz -dc validation.tar.gz | tar xf -


mamba activate env/


cd ${SCRATCHDIR}

python validation_cpu.py --limit 10000 --cpu 28 --batch 32

echo "All done"

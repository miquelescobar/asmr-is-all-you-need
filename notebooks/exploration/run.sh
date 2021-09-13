#!/bin/bash

#SBATCH --job-name=asmr-whis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 2   # Number of CPUs per task
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=19G           # CPU memory per node

pwd; hostname; date

set | grep SLURM

python3 wavenet-training.py
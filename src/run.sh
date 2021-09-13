#!/bin/bash

#BATCH --job-name=gpu30
#SBATCH --gres=gpu:1
#SBATCH --output=genv2_gpu_%j.log

pwd; hostname; date

set | grep SLURM

python3 Generate_sample.py

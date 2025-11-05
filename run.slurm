#!/bin/bash

#SBATCH --job-name=hyperSteer 

#SBATCH --partition=long # OPTIONS: long (2d) || medium (2h) || short (15m) || cpu
#SBATCH --time=2-00:00:00

#SBATCH --gres=gpu:1 # OPTIONS: gpu:1 || gpu:a100.80:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=out.log
##SBATCH --dependency=afterok:66153

# Load modules
module load CUDA #/12.5

# Activate virtual environment
source .venv/bin/activate

# Run script
python3 experiments/steering_generalization/run_steering_experiment.py
#!/bin/bash

# Initialize the module command first
source /etc/profile

# Load Conda environment
conda activate active-learning-env

echo "My run number: " $1

# Call your script as you would from the command line passing $1 and $2 as arguments
python PL_runner_ibmq_baseline_sim_long.py --run_seed=$1



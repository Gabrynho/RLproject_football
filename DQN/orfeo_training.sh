#!/bin/bash
#SBATCH -A dssc
#SBATCH -p GPU
#SBATCH --job-name=RL_DQN
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=2:0:0
#SBATCH --gres=gpu:1

python3 -m pip install . # Configuration of GFootball
python3 /DQN/trainAgent.py # Training of the DQN agent
#!/bin/bash
#SBATCH --job-name=smiles-lstm
#SBATCH --output=smiles-lstm.output
#SBATCH --time=0-00:10:00
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:volta:1

hostname
module purge; module load anaconda
source activate smiles-lstm-env
python experiments/03_train_model_locally.py --output "output/run_supercloud/"
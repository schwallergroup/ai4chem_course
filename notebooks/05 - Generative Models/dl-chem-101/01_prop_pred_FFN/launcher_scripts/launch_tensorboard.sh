#!/bin/bash
#SBATCH -n 1           # 1 core
#SBATCH -t 0-05:00:00   # 5 hours
#SBATCH -J ms # sensible name for the job
#SBATCH --output=logs/tensorboard%j.log   # Standard output and error log
#SBATCH -p sched_mit_ccoley
#SBATCH --mem 20G
#SBATCH -w node1238

##SBATCH -w node1238
##SBATCH --mem=20000 # 10 gb
##SBATCH --mem=20000 # 20 gb

##SBATCH --gres=gpu:1 #1 gpu
##SBATCH --mem=20000  # 20 gb 
##SBATCH -p {Partition Name} # Partition with GPUs

# Use this to run generic scripts:
# sbatch --export=MODEL_DIR="results/" launcher_scripts/launch_tensorboard.sh

# To access this, run the command:
# Pull the exact ip and port number frmo the log file after launch
# ssh -F ~/eofe-cluster/linux/config -L 6006:10.1.12.36:45316 eofe7.mit.edu
# Then navigate to localhost:6006

# Import module
source /etc/profile 
source /home/samlg/.bashrc

# Activate conda
# source {path}/miniconda3/etc/profile.d/conda.sh

# Activate right python version
# conda activate {conda_env}
conda activate ms-gen

# Evaluate the passed in command... in this case, it should be python
let ipnport=($UID-6025)%65274
echo ipnport=$ipnport

ipnip=$(hostname -i)
echo ipnip=$ipnip


tensorboard --logdir="${MODEL_DIR}" --port=$ipnport --bind_all


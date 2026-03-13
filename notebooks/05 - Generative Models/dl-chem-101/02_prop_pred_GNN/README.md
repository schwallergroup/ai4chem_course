# GNN Property Prediction

The repository uses a GNN applied to molecule graphs to predict molecular properties.

## Install

Install the conda enviornment:
```
conda env create -f enviornment.yml
pip install -r requirements.txt
```

To install this module for development:

```
python setup.py develop
```

Installing and structuring a package as a module makes it easy to avoid relative imports in code, which tend to complicate development.

## Training a model

While the package files contained inside `src/` should have the bulk of code, these files should not contain methods runnable directly. Instead, it's cleaner to have run scripts located outside the package file. Here, we use `run_scripts/` to hold these different files for executing code inside the package.


A simple model can be trained by running the command: 
```
python run_scripts/train_gnn.py
```

It can also be trained with gpu by simply adding a gpu flag (with a small datset and small model, this is runnable on cpu):

```
python run_scripts/train_gnn.py --gpu
```

To view all arguments with argparse: 

```
python run_scripts/train_gnn.py -h
```

## Using the trained model

### Making predictions

The trained model can be loaded directly from a TorchLightning checkpoint file and used to make predictions. Here, we can use our trained model to make predictions on the Enamine dataset of smiles strings alone.

```
python3 run_scripts/make_preds.py --smiles-file data/Enamine10k.csv
--checkpoint-pth results/example_run/version_0/epoch\=10-val_loss\=0.47.ckpt
--save-name results/example_run/preds.tsv --gpu
```
Again, this can also be made to run on gpu by adding `--gpu` at the end of the call.

### Analyzing results

Rather than maintain analysis scripts in the package itself. You can use a separate folder to contain any simple scripts for analyzing outputs. Here, we suggest using `analysis/` to hold such scripts. 

As an example, we provide a script to print the top k predictions:

```
python analysis/get_top_smiles.py  --input-file  results/example_run/preds.tsv
```

## Launching experiments with config files

While it's great to run scripts one at at time and get results, we often want to launch several experiments, take a break (lunch, beach, week-long vacation, etc.), and come back to get results. Launching these should be done programatically and consistently. We provide an example launcher script that shows how to build grids of different parameter inputs for a given python script, generate commands, and launch these in one of three settings:
1. Slurm  
2. Local  
3. Local\_parallel   

As an example, say we want to sweep over 3 different hidden parameter sizes:

```
python launcher_scripts/run_from_config.py configs/2022_08_07_exp_config_debug.yaml
```

The config file provides a list of hidden sizes, and program iteratiosn with each are launched. 

### Analysis

Once again, we can write a single analysis script to collect the results once it's done and view the test loss for each of these. 

```
python analysis/collect_hidden_sweep.py
```

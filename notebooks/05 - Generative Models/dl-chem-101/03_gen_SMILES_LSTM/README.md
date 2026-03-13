# Generative SMILES LSTM
An example of a deep molecular generative model based on a SMILES LSTM, to be used as a stand-alone package.

The code in this package is structured as follows:
* `src/` contains all the source code for the SMILES LSTM
* `scripts/` contains example scripts for downloading data, running training jobs, and visualizing the results of a training job
* `data/` is a placeholder directory for holding training data
* `output/` is a placeholder directory for the output from training jobs
* `analysis/` is a placeholder directory for figures and results of any data analyses

## Walkthrough
For a nice guide explaining the ins and outs of how a SMILES LSTM works, check out [SMILES-LSTM-Walkthrough.ipynb](./SMILES-LSTM-Walkthrough.ipynb)!

# Instructions
## 00 - Setting up the environment
Note: to use conda on molgpu01, you will need to install Miniconda locally. To do this, run:
```
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh
bash Miniconda3-py39_4.12.0-Linux-x86_64.sh 
```

To create the environment from scratch using conda, run:
```
conda create -n smiles-lstm-env pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
conda activate smiles-lstm-env
pip install rdkit
pip install PyTDC
```

We will then install the SMILES LSTM as a package using the provided `setup.py`. With the environment activated, run the following command from the `03_gen_SMILES_LSTM/` directory:
```
pip install -e .
```

## 01 - Downloading the data
To train the SMILES LSTM, we will be downloading train/test/valid split data from the Therapeutics Data Commons (TDC). We will use ZINC. To do this, run:
```
python ./scripts/01_download_data.py
```

The script will divide the available ZINC data into a 50K/5K/5K set of train/test/valid splits.

## 02 - Training the SMILES LSTM locally
To train the model on a local GPU, run:
```
python ./scripts/02_train_model_locally.py
```

The model will train for 10 epochs on 50K SMILES sampled from the aforementioned ZINC dataset. This will take about 5 minutes on a single GPU. After 10 epochs, the model will not be fully trained but generating ~20% valid SMILES. You can see what the molecules being generated look like after each epoch N by opening the `sampled_epoch{N}.png` files generated within each job directory.

## 03 - Training the SMILES LSTM on a high-performance computing cluster (SuperCloud)
To train the model on the Supercloud, run:
```
python ./scripts/03_train_model_SuperCloud.py
```

This will submit the training job as a batch job using LLsub and run for about 5 minutes (timeout set to 10 minutes). It should produce similar results to the job run locally.

You can modify this script to run on any other cluster (you will most likely need to modify the argument `--gres=gpu:volta:1` depending on the type of GPUs on the other cluster and how it is configured).

## 04 - Analysis and making figures
As an example, we can visualize how the training and validation losses change during training, as well as how the validity of molecules sampled from our SMILES LSTM increases as we train it for more epochs. To visualize these results, run:
```
python ./scripts/04_plot_results.py --jobdir ./output/run_local/
``` 

This will create two plots in the `./analysis/` directory.

To visualize results for a different job, simply replace the argument after `--output` to the path to your new job and rerun.

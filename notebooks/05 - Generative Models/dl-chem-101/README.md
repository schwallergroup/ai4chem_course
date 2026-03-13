# dl-chem-101

A repository for various introductory tutorials on deep learning for chemistry.

* [FFN Property Prediction](./01_prop_pred_FFN/)
* [GNN Property Prediction](./02_prop_pred_GNN/)
* [LSTM SMILES Generation](./03_gen_SMILES_LSTM/)

## Introduction

There is often a gap between code written in classes (computer science, chemistry, etc.) and
code required to conduct research. Most classes now support Jupyter Notebooks or Google Colab
enviornments that have simple install, setup, and often require running only small blocks of code.
While very useful and didactic, we find there is also a need to explain how students 
can structure repositories for new research projects that enable them to organize
experiments, try different model settings, and move quickly.

This repository is an opinionated attempt to show several ways to structure
these repositories for basic tasks we expect any researcher at the intersection of
machine learning and chemistry to implement. Specifically:

1. Molecular property prediction with feed forward networks   
2. Molecular property prediction with graph neural networks  
3. Molecular generation with a SMILES LSTM  

### How should I use this? 

We recommend two ways to use this repository:
1. Reattempting tasks 

Consider attempting the tasks described from scratch and compare to how we've done it.

2. Adding documentation

We recognize that attempting these may be too time consuming for shorter
onboarding periods. As an alternative, we provide versions of the code with _no
documentation_ at [github.com/coleygroup/dl-chem-101-stripped](https://github.com/coleygroup/dl-chem-101-stripped). As a useful exercise, 
we recommend forking [the repo](https://github.com/coleygroup/dl-chem-101-stripped), running the code, and then
adding documentation to each function (i.e., docstrings). Such docstrings should specify: 
1. What the function / class does  
2. The type and shape of the inputs and outputs  
3. Any complex details within the function using inline comments  

### Learning outcomes

1. How to structure an ML-for-chemistry repository
2. How to launch experiments for various model parameters and configurations
3. How to separate analysis from model predictions

## Problem Prompts

For those interested in attempting this on their own before viewing our
solutions and structures, we provide the following guiding prompts and
references.

### 01_prop_pred_FFN
In this repository, we will use a feed-forward neural network (FFN) to predict a molecular property relevant to drug discovery, Caco-2 cell permeability, from [molecular fingerprints](https://doi.org/10.1021/ci100050t) (originally demonstrated in [Wang *et al.* (2016)](https://doi.org/10.1021/acs.jcim.5b00642)). 

We use data available for download via the [Therapeutics Data Commons (TDC)](https://tdcommons.ai/single_pred_tasks/adme/#caco-2-cell-effective-permeability-wang-et-al) (original paper introducing the TDC from [Huang *et al.* (2021)](https://arxiv.org/abs/2102.09548)).


### 02_prop_pred_GNN

This repository repeats the above task but utilizes graph neural networks that operaate on molecular graphs directly, rather than vectorized fingerprints.

Some foundational papers in graph neural network development for property prediction are [Gilmer *et al.* (2017)](https://proceedings.mlr.press/v70/gilmer17a.html) and [Duvenaud *et al.* (2015)](https://proceedings.neurips.cc/paper/2015/hash/f9be311e65d81a9ad8150a60844bb94c-Abstract.html). 

Several groups have compared performance between graph and fingerprint-based neural networks (i.e, MoleculeNet ([Wu *et al.* (2017)](https://pubs.rsc.org/en/content/articlehtml/2018/sc/c7sc02664a#cit63)) and ChemProp ([Yang *et al.* (2019)](https://doi.org/10.1021/acs.jcim.9b00237)))

### 03_gen_smiles_LSTM
In this repository, we will go through the process of training a SMILES long short-term memory (LSTM) network for molecular design tasks. At a high level, the model "sees" examples of valid molecular SMILES strings, and learns to generate new strings from the same distribution by progressively predicting the next token in the string. These models have a long history in natural language processing, in which context neural networks are trained to complete sentences when given a set of starting words.

We recommend reviewing both [Segler *et al.* (2018)](https://doi.org/10.1021/acscentsci.7b00512) and [Bjerrum, E. J. (2017)](https://arxiv.org/abs/1703.07076), two of the earliest examples of such models.

Code for this example was adapted from the SMILES LSTM implementation in the Molecular AI [REINVENT](https://github.com/MolecularAI/Reinvent) repository and structured as a stand-alone package.

Here, we train only on a smaller 50K subset of SMILES strings from the ZINC dataset [available from the TDC](https://tdcommons.ai/generation_tasks/molgen/). We also show how to run our model training script both on a local GPU and on an MIT/Lincoln Lab specific cluster, SuperCloud (using the Slurm-based LLSub system).

## Authors
* [Samuel Goldman](https://github.com/samgoldman97)
* [Rocío Mercado](https://github.com/rociomer)

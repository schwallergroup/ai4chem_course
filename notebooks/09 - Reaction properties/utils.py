# https://github.com/pschwllr/MolecularTransformer/blob/master/score_predictions.py

import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdchiral.main import rdchiralReaction, rdchiralReactants, rdchiralRun


def load_data():
    train = pd.read_csv('data/raw_train.csv')[['reactants>reagents>production']]
    val = pd.read_csv('data/raw_val.csv')[['reactants>reagents>production']]
    test = pd.read_csv('data/raw_test.csv')[['reactants>reagents>production']]
    
    return train, val, test
    
    
    
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.molSize = (800, 200)
IPythonConsole.highlightByReactant = True

def visualize_chemical_reaction(reaction_smarts: str):
    # Create a RDKit reaction object from reaction SMARTS string
    reaction = AllChem.ReactionFromSmarts(reaction_smarts, useSmiles=True)
    # Display images
    display(reaction)
    
    
def visualize_mols(mol_smi):
    mol = Chem.MolFromSmiles(mol_smi)
    img = Draw.MolToImage(mol)
    display(img)
    
    
from rxnutils.chem.reaction import ChemicalReaction
def extract_template(rxn):

    rxn = ChemicalReaction(rxn)
    rxn_smarts = rxn.generate_reaction_template(radius=0)
    if type(rxn_smarts)==tuple:
        return rxn_smarts[0].smarts
    return rxn_smarts.smarts

def apply_template(tmplt, reacts):
    rxn = AllChem.ReactionFromSmarts(tmplt)
    reactants = [Chem.MolFromSmiles(x) for x in reacts.split('.')]
    products = rxn.RunReactants(reactants)
    return products


import scipy
from scipy import sparse
from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


def mol_smi_to_count_fp(
    mol_smi: str, radius: int = 2, fp_size: int = 32681, dtype: str = "int32"
):
    """
    taken from https://github.com/linminhtoo/neuralsym/blob/d17810acc497b15252664f7700733d969815e5ec/prepare_data.py
    """
    
    fp_gen = GetMorganGenerator(
        radius=radius, includeChirality=True, fpSize=fp_size
    )
    mol = Chem.MolFromSmiles(mol_smi)
    uint_count_fp = fp_gen.GetCountFingerprint(mol)
    count_fp = np.empty((1, fp_size), dtype=dtype)
    DataStructs.ConvertToNumpyArray(uint_count_fp, count_fp)
    return sparse.csr_matrix(count_fp, dtype=dtype)
            
    
    
import re
def canonicalize_smiles(smiles, verbose=False): # will raise an Exception if invalid SMILES
        
    # First clean aam numbers
    smiles = re.sub(r"(?<=[^\*])(:\d+)]", "]", smiles)
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol)
    else:
        if verbose:
            print(f'{smiles} is invalid.')
        return ''

def get_eval_df():
    n_best = 5 # top-5 predictions were outputted
    predictions = [[] for i in range(n_best)]

    with open('USPTO_480k_preprocessed/products-val.txt', 'r') as f:
        targets = [line.strip().replace(' ', '') for line in f]

    evaluation_df = pd.DataFrame(targets)
    evaluation_df.columns = ['target']

    with open('USPTO_480k_preprocessed/precursors-val.txt', 'r') as f:
        precursors = [line.strip().replace(' ', '') for line in f]
    evaluation_df['precursors'] = precursors

    total = len(evaluation_df)

    with open('models/USPTO480k_model_step_400000_val_predictions.txt', 'r') as f:

        for i, line in enumerate(f):
            predictions[i % n_best].append(''.join(line.strip().split(' ')))
    for i, preds in enumerate(predictions):
        evaluation_df['prediction_{}'.format(i + 1)] = preds
        evaluation_df['canonical_prediction_{}'.format(i + 1)] = evaluation_df['prediction_{}'.format(i + 1)].progress_apply(
            lambda x: canonicalize_smiles(x)
        )
    return evaluation_df
    
    
def get_prediction_rank(row, col_name, max_rank):
    for i in range(1, max_rank+1):
        if row['target'] == row['{}{}'.format(col_name, i)]:
            return i
    return 0

def evaluate(n_best):

    evaluation_df = get_eval_df()
    total = evaluation_df.shape[0]
    evaluation_df['prediction_rank'] = evaluation_df.progress_apply(lambda row: get_prediction_rank(row, 'canonical_prediction_', n_best), axis=1)

    correct = 0

    for i in range(1, n_best+1):
        correct += (evaluation_df['prediction_rank'] == i).sum()
        invalid_smiles = (evaluation_df['canonical_prediction_{}'.format(i)] == '').sum()

        print('Top-{}: {:.1f}% || Invalid SMILES {:.2f}%'.format(i, correct/total*100,
                                                                     invalid_smiles/total*100))

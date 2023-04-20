# https://github.com/pschwllr/MolecularTransformer/blob/master/score_predictions.py

import os
import gdown
import pandas as pd
from rdkit import Chem


def download_data():
    # links from https://github.com/coleygroup/Graph2SMILES/blob/main/scripts/download_raw_data.py
    USPTO_480k_links= [
            ("https://drive.google.com/uc?id=1RysNBvB2rsMP0Ap9XXi02XiiZkEXCrA8", "src-train.txt"),
            ("https://drive.google.com/uc?id=1CxxcVqtmOmHE2nhmqPFA6bilavzpcIlb", "tgt-train.txt"),
            ("https://drive.google.com/uc?id=1FFN1nz2yB4VwrpWaBuiBDzFzdX3ONBsy", "src-val.txt"),
            ("https://drive.google.com/uc?id=1pYCjWkYvgp1ZQ78EKQBArOvt_2P1KnmI", "tgt-val.txt"),
            ("https://drive.google.com/uc?id=10t6pHj9yR8Tp3kDvG0KMHl7Bt_TUbQ8W", "src-test.txt"),
            ("https://drive.google.com/uc?id=1FeGuiGuz0chVBRgePMu0pGJA4FVReA-b", "tgt-test.txt")
        ]
    data_path = 'USPTO_480k'
    os.makedirs(data_path, exist_ok=True)
    for url, name in USPTO_480k_links:
        target_path = os.path.join(data_path, name)
        if not os.path.exists(target_path):
            gdown.download(url, target_path, quiet=False)
        else:
            print(f"{target_path} already exists")
            
    with open('USPTO_480k/src-train.txt', 'r') as f:
        precursors_train = [line.strip().replace(' ', '') for line in f]
    with open('USPTO_480k/tgt-train.txt', 'r') as f:
        products_train = [line.strip().replace(' ', '') for line in f]
    with open('USPTO_480k/src-val.txt', 'r') as f:
        precursors_val = [line.strip().replace(' ', '') for line in f]
    with open('USPTO_480k/tgt-val.txt', 'r') as f:
        products_val = [line.strip().replace(' ', '') for line in f]
    with open('USPTO_480k/src-test.txt', 'r') as f:
        precursors_test = [line.strip().replace(' ', '') for line in f]
    with open('USPTO_480k/tgt-test.txt', 'r') as f:
        products_test = [line.strip().replace(' ', '') for line in f]
    
    train_df = pd.DataFrame({'precursors': precursors_train, 'products': products_train})
    print(f"The training set contains {train_df.shape[0]} reactions.")
    train_df.head()

    test_df = pd.DataFrame({'precursors': precursors_test, 'products': products_test})
    print(f"The testing set contains {test_df.shape[0]} reactions.")
    test_df.head()

    val_df = pd.DataFrame({'precursors': precursors_val, 'products': products_val})
    print(f"The validation set contains {val_df.shape[0]} reactions.")
    val_df.head()
    
    return train_df, val_df, test_df

            
def canonicalize_smiles(smiles, verbose=False): # will raise an Exception if invalid SMILES
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

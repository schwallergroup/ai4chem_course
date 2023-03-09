
def load_esol_data():

    import pandas as pd

    # load dataset from the CSV file
    esol_df = pd.read_csv('data/esol.csv')

    # Get NumPy arrays from DataFrame for the input and target
    smiles = esol_df['smiles'].values
    y = esol_df['log solubility (mol/L)'].values

    # Here, we use molecular descriptors from RDKit, like molecular weight, number of valence electrons, maximum and minimum partial charge, etc.
    from deepchem.feat import RDKitDescriptors
    featurizer = RDKitDescriptors()
    features = featurizer.featurize(smiles)
    print(f"Number of generated molecular descriptors: {features.shape[1]}")

    # Drop the features containing invalid values
    import numpy as np
    features = features[:, ~np.isnan(features).any(axis=0)]
    print(f"Number of molecular descriptors without invalid values: {features.shape[1]}")


    # Data preprocessing
    from sklearn.model_selection import train_test_split
    X = features
    # training data size : test data size = 0.8 : 0.2
    # fixed seed using the random_state parameter, so it always has the same split.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=0)

    # Create a validation set from the train set
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, train_size=0.8, random_state=0)


    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(X_train)

    # save original X
    X_train_ori = X_train
    X_test_ori = X_test
    # transform data
    X_train = scaler.transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    return X_train, X_valid, X_test, y_train, y_valid, y_test, scaler

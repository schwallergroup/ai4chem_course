import torch
from smiles_lstm.model.smiles_vocabulary import SMILESTokenizer
from smiles_lstm.model.smiles_lstm import SmilesLSTM

def load_from_file(file_path : str, sampling_mode : bool=False):
    """
    Loads a model from a single file.
    Params:
    ------
        file_path (str) : Input file path.
    Returns:
    -------
        SmilesLSTM : New instance of the RNN, or an exception if it was not
                     possible to load it.
    """
    model = torch.load(file_path, map_location='cpu')
    if sampling_mode:
        model.network.eval()

    return model

"""
Functions for loading molecules from SMILES.
"""
import pandas as pd
import rdkit
from rdkit.Chem.rdmolfiles import SmilesMolSupplier


def molecules(path : str) -> rdkit.Chem.rdmolfiles.SmilesMolSupplier:
    """
    Reads a SMILES file (full path/filename specified by `path`) and returns
    `rdkit.Mol` objects.
    """
    # check first line of SMILES file to see if contains header
    with open(path, "r", encoding="utf-8") as smi_file:
        first_line = smi_file.readline()
        has_header = bool("SMILES" in first_line)
    smi_file.close()

    # read file
    molecule_set = SmilesMolSupplier(path,
                                     sanitize=True,
                                     nameColumn=-1,
                                     titleLine=has_header)
    return molecule_set

def smiles(path : str) -> list:
    """
    Reads a SMILES file (full path/filename specified by `path`) and returns
    SMILES (`str`) objects.
    """
    smiles_list = pd.read_csv(path, delimiter=" ")["SMILES"].to_list()
    return smiles_list

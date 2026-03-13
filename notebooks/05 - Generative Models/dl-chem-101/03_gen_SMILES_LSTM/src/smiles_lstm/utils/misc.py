"""
Miscellaneous utilities.
"""
import csv
from builtins import ZeroDivisionError
from warnings import filterwarnings
from PIL import Image
import rdkit
from rdkit import RDLogger, Chem
from rdkit.Chem import Draw
import torch
import tqdm


def suppress_warnings(level : str="minor") -> None:
    """
    Suppresses unimportant warnings for a cleaner readout.
    Args:
    ----
        level (str) : The level of warnings to suppress. Options: "minor" or
                      "all". Defaults to "minor".
    """
    if level == "minor":
        RDLogger.logger().setLevel(RDLogger.CRITICAL)
        filterwarnings(action="ignore", category=UserWarning)
        filterwarnings(action="ignore", category=FutureWarning)
    elif level == "all":
        # instead suppress ALL warnings
        filterwarnings(action="ignore")
    else:
        raise ValueError(f"Not a valid `level`. Use 'minor' or 'all', not '{level}'.")

def get_device() -> str:
    """
    Gets the available device (GPU or CPU).
    """
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return device

def draw_smiles(path : str, smiles_list : list) -> float:
    """
    Save an image of the molecules sampled (from a PIL image) to the specified
    path, as well as the fraction valid.
    """
    mols          = []
    smiles_legend = []
    n_valid       = 0
    for smiles in smiles_list:
        try:
            mol = rdkit.Chem.MolFromSmiles(smiles, sanitize=False)
            mol.UpdatePropertyCache(strict=True)
            rdkit.Chem.Kekulize(mol)
            if mol is not None:
                mols.append(mol)
                smiles_legend.append(smiles)
                n_valid += 1
        except:
            # don't draw invalid molecules
            pass

    # compute the fraction valid, which will be returned
    try:
        fraction_valid = n_valid/len(smiles_list)
    except ZeroDivisionError:
        fraction_valid = 0.0

    try:
        # sketch the sampled molecules
        image = rdkit.Chem.Draw.MolsToGridImage(mols,
                                                molsPerRow=4,
                                                subImgSize=(300, 300),
                                                legends=smiles_legend,
                                                highlightAtomLists=None)
        image.save(path, format='png')
    except:
        # blank PIL image placeholder if something was wrong in above sketch
        image = Image.new('RGB', (300, 300))
        image.save(path, format='png')

    return fraction_valid

def progress_bar(iterable : iter, total : int, **kwargs) -> tqdm.tqdm:
    """
    Return a tqdm (progress bar) object for an iterable.

    Params:
    ------
        iterable (iter) : Object to create progress bar for.
        total (int)     : Total number of items in the iterable.

    Returns:
    -------
        tqdm.tqdm : Progress bar.
    """
    return tqdm.tqdm(iterable=iterable, total=total, ascii=True, **kwargs)

def save_smiles(smiles : list, output_filename : str) -> None:
    """
    Saves the generated SMILES for the current step to a CSV.
    
    Params:
    ------
        smiles (list)         : Contains the SMILES to save.
        output_filename (str) : Name of file to which to save SMILES to.
    """
    with open(output_filename, "w", encoding="utf-8") as output_file:

        # using csv.writer method from CSV package
        write = csv.writer(output_file, delimiter="\n")  # write each SMILES on a new line
        write.writerow(smiles)

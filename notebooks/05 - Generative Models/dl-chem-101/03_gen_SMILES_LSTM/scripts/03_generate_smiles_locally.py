"""
Generate SMILES from a trained SMILES LSTM model and visualize them.
"""
import argparse
from pathlib import Path
import torch
import numpy as np

from smiles_lstm.model.smiles_lstm import SmilesLSTM
from smiles_lstm.utils.misc import suppress_warnings, draw_smiles, save_smiles, get_device

# suppress minor warnings
suppress_warnings()

# define the argument parser
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    add_help=False,
    description="Generate and visualize SMILES from a trained SMILES LSTM."
)

parser.add_argument("--model_path",
                    type=str,
                    required=True,
                    help="Path to the trained model checkpoint (.pth file).")
parser.add_argument("--output_dir",
                    type=str,
                    default="./output/generate/",
                    help="Specifies the path to save generated SMILES and images.")
parser.add_argument("--n_samples",
                    type=int,
                    default=1000,
                    help="Number of SMILES to generate.")
parser.add_argument("--visualize",
                    type=int,
                    default=25,
                    help="Number of valid molecules to visualize in a grid (0 to skip visualization).")
parser.add_argument("--seed",
                    type=int,
                    default=42,
                    help="Random seed for reproducibility.")

args = parser.parse_args()

if __name__ == '__main__':
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("*** Loading trained model ***", flush=True)
    model_path = Path(args.model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load the trained model using the class method
    smiles_lstm = SmilesLSTM.load_from_file(str(model_path), sampling_mode=True)
    
    print(f"Loaded model from: {model_path}")
    print(f"Using device: {get_device()}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n*** Generating {args.n_samples} SMILES ***", flush=True)
    
    # Generate SMILES using the sample_smiles method
    all_smiles, nlls = smiles_lstm.sample_smiles(
        num=args.n_samples,
        batch_size=512,
    )
    
    # Save ALL generated SMILES (raw output, no filtering)
    output_file = output_dir / "generated_smiles.smi"
    save_smiles(smiles=all_smiles, output_filename=str(output_file))
    print(f"\nSaved {len(all_smiles)} generated SMILES to: {output_file}")
    
    # Compute validity statistics (for reporting only)
    from rdkit import Chem
    
    valid_smiles = []
    for smi in all_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_smiles.append(smi)
    
    unique_valid_smiles = list(dict.fromkeys(valid_smiles))
    
    print(f"\n*** Generation Summary ***")
    print(f"  Total generated: {len(all_smiles)}")
    print(f"  Valid: {len(valid_smiles)} ({100*len(valid_smiles)/len(all_smiles):.1f}%)")
    print(f"  Invalid: {len(all_smiles) - len(valid_smiles)} ({100*(len(all_smiles) - len(valid_smiles))/len(all_smiles):.1f}%)")
    print(f"  Unique (valid): {len(unique_valid_smiles)} ({100*len(unique_valid_smiles)/len(valid_smiles):.1f}% of valid)")
    
    # Visualize valid molecules using draw_smiles utility
    if args.visualize > 0 and len(valid_smiles) > 0:
        print(f"\n*** Visualizing {min(args.visualize, len(valid_smiles))} valid molecules ***", flush=True)
        
        smiles_to_draw = valid_smiles[:args.visualize]
        img_path = output_dir / "generated_molecules.png"
        
        # draw_smiles returns the fraction valid
        fraction_valid = draw_smiles(
            path=str(img_path),
            smiles_list=smiles_to_draw
        )
        
        print(f"Saved molecule grid to: {img_path}")
    
    print("\n*** Generation complete ***")
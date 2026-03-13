"""
Fine-tune a pre-trained SMILES LSTM on a new dataset (e.g., DRD2 actives).
Loads a pre-trained model and continues training with a new dataset.
"""
import argparse
from pathlib import Path
import torch

from smiles_lstm.model.smiles_lstm import SmilesLSTM
from smiles_lstm.model.smiles_trainer import SmilesTrainer
from smiles_lstm.model.smiles_vocabulary import SMILESTokenizer
from smiles_lstm.utils import load
from smiles_lstm.utils.misc import suppress_warnings

# suppress minor warnings
suppress_warnings()

# define the argument parser
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    add_help=False,
    description="Fine-tune a pre-trained SMILES LSTM on a new dataset."
)

parser.add_argument("--pretrained_model",
                    type=str,
                    required=True,
                    help="Path to the pre-trained model checkpoint (.pth file).")
parser.add_argument("--data_dir",
                    type=str,
                    default="./data/drd2_finetune/",
                    help="Specifies the path to the fine-tuning data.")
parser.add_argument("--output_dir",
                    type=str,
                    default="./output/finetune/",
                    help="Specifies the path to which to write output to.")
parser.add_argument("--epochs",
                    type=int,
                    default=10,
                    help="Specifies how many epochs to fine-tune for.")
parser.add_argument("--learning_rate",
                    type=float,
                    default=0.0001,
                    help="Learning rate for fine-tuning (typically lower than pre-training).")
parser.add_argument("--samples",
                    type=int,
                    default=256,
                    help="Specifies how many samples to generate each evaluation epoch.")

args = parser.parse_args()


def filter_smiles_by_vocabulary(smiles_list, vocabulary, tokenizer):
    """
    Filter SMILES to only include those whose tokens are all in the vocabulary.
    
    Args:
        smiles_list: List of SMILES strings
        vocabulary: The vocabulary from the pre-trained model
        tokenizer: The tokenizer to use
        
    Returns:
        filtered_smiles: List of SMILES that only contain known tokens
        excluded_smiles: List of SMILES that were excluded
        unknown_tokens: Set of tokens that were not in the vocabulary
    """
    filtered_smiles = []
    excluded_smiles = []
    unknown_tokens = set()
    
    # Get the set of known tokens from the vocabulary
    known_tokens = set(vocabulary._tokens.keys())
    
    for smi in smiles_list:
        tokens = tokenizer.tokenize(smi)
        unknown_in_smi = [t for t in tokens if t not in known_tokens]
        
        if len(unknown_in_smi) == 0:
            filtered_smiles.append(smi)
        else:
            excluded_smiles.append(smi)
            unknown_tokens.update(unknown_in_smi)
    
    return filtered_smiles, excluded_smiles, unknown_tokens


if __name__ == '__main__':
    print("*** Loading pre-trained model ***", flush=True)
    pretrained_path = Path(args.pretrained_model)
    
    if not pretrained_path.exists():
        raise FileNotFoundError(f"Pre-trained model not found: {pretrained_path}")
    
    # Load the pre-trained model using the class method
    smiles_lstm = SmilesLSTM.load_from_file(str(pretrained_path), sampling_mode=False)
    
    print(f"Loaded model from: {pretrained_path}")
    print(f"Vocabulary size: {len(smiles_lstm.vocabulary)}")
    
    print("\n*** Loading fine-tuning data ***", flush=True)
    data_path = Path(args.data_dir)
    
    train_raw = load.smiles(path=(data_path.joinpath("train.smi")))
    valid_raw = load.smiles(path=(data_path.joinpath("valid.smi")))
    
    # For fine-tuning, test set is optional
    test_path = data_path.joinpath("test.smi")
    if test_path.exists():
        test_raw = load.smiles(path=test_path)
    else:
        test_raw = valid_raw  # Use validation as test if no test set provided
    
    print(f"Loaded data: {len(train_raw)} train, {len(valid_raw)} valid, {len(test_raw)} test")
    
    # Filter SMILES to only include those compatible with pre-trained vocabulary
    print("\n*** Filtering SMILES by pre-trained vocabulary ***", flush=True)
    
    tokenizer = smiles_lstm.tokenizer
    vocabulary = smiles_lstm.vocabulary
    
    train, train_excluded, train_unknown = filter_smiles_by_vocabulary(train_raw, vocabulary, tokenizer)
    valid, valid_excluded, valid_unknown = filter_smiles_by_vocabulary(valid_raw, vocabulary, tokenizer)
    test, test_excluded, test_unknown = filter_smiles_by_vocabulary(test_raw, vocabulary, tokenizer)
    
    all_unknown = train_unknown | valid_unknown | test_unknown
    
    print(f"Training:   {len(train)}/{len(train_raw)} kept ({len(train_excluded)} excluded)")
    print(f"Validation: {len(valid)}/{len(valid_raw)} kept ({len(valid_excluded)} excluded)")
    print(f"Test:       {len(test)}/{len(test_raw)} kept ({len(test_excluded)} excluded)")
    
    if all_unknown:
        print(f"\nUnknown tokens found: {sorted(all_unknown)}")
    
    if len(train) == 0:
        raise ValueError("No training samples remaining after filtering! Check your data.")
    
    print(f"\nFine-tuning data after filtering: {len(train)} train, {len(valid)} valid, {len(test)} test")
    
    # Trainer object expects data in a dictionary
    SMILES_dict = {"train": train, "valid": valid, "test": test}
    
    print("\n*** Creating Trainer object for fine-tuning ***", flush=True)
    trainer = SmilesTrainer(
        model=smiles_lstm,
        input_smiles=SMILES_dict,
        epochs=args.epochs,
        shuffle=True,
        batch_size=512,
        learning_rate=args.learning_rate,
        augment=0,
        output_model_path=args.output_dir,
        start_epoch=0,
        learning_rate_scheduler="StepLR",
        gamma=0.95,
        eval_num_samples=args.samples,
        eval_batch_size=256
    )
    
    print("\n*** Fine-tuning the model ***", flush=True)
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Learning rate: {args.learning_rate}")
    trainer.run()
    
    print("\n*** Fine-tuning complete ***", flush=True)
    print(f"Model saved to: {args.output_dir}")
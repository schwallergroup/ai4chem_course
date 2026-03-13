"""predict.py

Make predictions with trained model

"""
import logging
import yaml
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from pred_ffn import ffn_data, ffn_model, utils


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--gpu", default=False, action="store_true")
    parser.add_argument("--num-workers", default=0, action="store", type=int)
    parser.add_argument("--batch-size", default=64, action="store", type=int)
    parser.add_argument("--save-name", default="results/example_run/preds.tsv")
    parser.add_argument(
        "--checkpoint-pth",
        help="name of checkpoint file",
        default="results/example_run/version_3/epoch=99-val_loss=1.56.ckpt")
    parser.add_argument("--smiles-file", default="data/Enamine10k.csv")
    return parser.parse_args()


def predict():
    args = get_args()
    kwargs = args.__dict__

    debug = kwargs['debug']
    kwargs['save_dir'] = str(Path(kwargs['save_name']).parent)
    save_dir = kwargs['save_dir']
    utils.setup_logger(save_dir,
                       log_name="gnn_pred.log",
                       debug=kwargs['debug'])
    pl.utilities.seed.seed_everything(kwargs.get("seed"))

    # Dump args
    yaml_args = yaml.dump(kwargs, indent=2, default_flow_style=False)
    logging.info(f"Args:\n{yaml_args}")
    with open(Path(save_dir) / "args.yaml", "w") as fp:
        fp.write(yaml_args)

    # Get dataset
    # Load smiles for pred
    input_smiles = Path(kwargs['smiles_file'])
    if not input_smiles.exists():
        raise ValueError(f"Unable to find file {input_smiles}")
    smiles = [i.strip() for i in open(input_smiles, "r").readlines()]

    if debug:
        smiles = smiles[:200]

    # Get train, val, test inds
    num_workers = kwargs.get("num_workers", 0)
    pred_dataset = ffn_data.MolDataset(smiles, num_workers=num_workers)

    # Define dataloaders
    collate_fn = pred_dataset.get_collate_fn()
    pred_loader = DataLoader(pred_dataset,
                             num_workers=kwargs['num_workers'],
                             collate_fn=collate_fn,
                             shuffle=False,
                             batch_size=kwargs['batch_size'])

    # Create model and load
    best_checkpoint = kwargs['checkpoint_pth']

    # Load from checkpoint
    model = ffn_model.ForwardFFN.load_from_checkpoint(best_checkpoint)
    logging.info(f"Loaded model with from {best_checkpoint}")

    model.eval()
    gpu = kwargs['gpu']
    if gpu:
        model = model.cuda()

    names, preds = [], []
    with torch.no_grad():
        for batch in pred_loader:
            fps, batch_names = batch['fps'], batch['names']
            if gpu:
                fps = fps.cuda()
            output = model(fps).cpu().detach().numpy()
            preds.append(output)
            names.append(batch_names)

        names = [j for i in names for j in i]
        preds = np.vstack(preds)

        output = {"preds": preds.squeeze(), "smiles": names}
        output = pd.DataFrame(output)
        output.to_csv(kwargs['save_name'], sep="\t", index=None)

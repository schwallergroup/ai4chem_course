"""train.py

Train ffn to predict binned specs

"""
import logging
import yaml
import argparse
from pathlib import Path
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from pred_ffn import utils, ffn_data, ffn_model

from tdc.single_pred import ADME


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--gpu", default=False, action="store_true")
    parser.add_argument("--seed", default=42, action="store", type=int)
    parser.add_argument("--num-workers", default=0, action="store", type=int)
    parser.add_argument("--batch-size", default=128, action="store", type=int)
    parser.add_argument("--max-epochs", default=100, action="store", type=int)
    parser.add_argument("--save-dir", default="results/example_run/")

    parser.add_argument("--dataset-name", default="caco", choices=["caco"])

    # Model args
    parser.add_argument("--layers", default=3, action="store", type=int)
    parser.add_argument("--dropout", default=0.1, action="store", type=float)
    parser.add_argument("--hidden-size", default=256, action="store", type=int)
    return parser.parse_args()


def train_model():
    args = get_args()
    kwargs = args.__dict__

    save_dir = kwargs['save_dir']
    utils.setup_logger(save_dir,
                       log_name="ffn_train.log",
                       debug=kwargs['debug'])
    pl.utilities.seed.seed_everything(kwargs.get("seed"))

    # Dump args
    yaml_args = yaml.dump(kwargs, indent=2, default_flow_style=False)
    with open(Path(save_dir) / "args.yaml", "w") as fp:
        fp.write(yaml_args)

    logging.info(f"Args:\n{yaml_args}")

    # Extract data
    if kwargs['dataset_name'] == "caco":
        data = ADME(name='Caco2_Wang')
        # df = data.get_data()
        splits = data.get_split()
        train_smi, train_y = zip(*splits['train'][['Drug', 'Y']].values)
        valid_smi, valid_y = zip(*splits['valid'][['Drug', 'Y']].values)
        test_smi, test_y = zip(*splits['test'][['Drug', 'Y']].values)
    else:
        raise NotImplementedError()

    num_workers = kwargs.get("num_workers", 0)
    train_dataset = ffn_data.PredDataset(
        train_smi,
        train_y,
        num_workers=num_workers,
    )
    valid_dataset = ffn_data.PredDataset(
        valid_smi,
        valid_y,
        num_workers=num_workers,
    )
    test_dataset = ffn_data.PredDataset(
        test_smi,
        test_y,
        num_workers=num_workers,
    )
    dataset_sizes = (len(train_dataset), len(valid_dataset), len(test_dataset))
    logging.info(f"Train, val, test sizes: {dataset_sizes}")

    # Define dataloaders
    collate_fn = train_dataset.get_collate_fn()
    train_loader = DataLoader(train_dataset,
                              num_workers=kwargs['num_workers'],
                              collate_fn=collate_fn,
                              shuffle=True,
                              batch_size=kwargs['batch_size'])
    val_loader = DataLoader(valid_dataset,
                            num_workers=kwargs['num_workers'],
                            collate_fn=collate_fn,
                            shuffle=False,
                            batch_size=kwargs['batch_size'])
    test_loader = DataLoader(test_dataset,
                             num_workers=kwargs['num_workers'],
                             collate_fn=collate_fn,
                             shuffle=False,
                             batch_size=kwargs['batch_size'])

    # Define model
    test_batch = next(iter(train_loader))

    model = ffn_model.ForwardFFN(
        hidden_size=kwargs['hidden_size'],
        layers=kwargs['layers'],
        dropout=kwargs['dropout'],
        output_dim=1,
    )

    # outputs = model(test_batch['fps'])

    # Create trainer
    tb_logger = pl_loggers.TensorBoardLogger(save_dir, name="")
    console_logger = utils.ConsoleLogger()

    tb_path = tb_logger.log_dir
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=tb_path,
        filename="{epoch}-{val_loss:.2f}",
        save_weights_only=False,
    )
    earlystop_callback = EarlyStopping(monitor="val_loss", patience=10)
    callbacks = [earlystop_callback, checkpoint_callback]

    trainer = pl.Trainer(logger=[tb_logger, console_logger],
                         accelerator="gpu" if kwargs['gpu'] else "cpu",
                         gpus=1 if kwargs['gpu'] else 0,
                         callbacks=callbacks,
                         gradient_clip_val=5,
                         max_epochs=kwargs['max_epochs'],
                         gradient_clip_algorithm="value")

    trainer.fit(model, train_loader, val_loader)
    checkpoint_callback = trainer.checkpoint_callback
    best_checkpoint = checkpoint_callback.best_model_path
    best_checkpoint_score = checkpoint_callback.best_model_score.item()

    # Load from checkpoint
    model = ffn_model.ForwardFFN.load_from_checkpoint(best_checkpoint)
    logging.info(
        f"Loaded model with from {best_checkpoint} with val loss of {best_checkpoint_score}"
    )

    model.eval()
    test_out = trainer.test(dataloaders=test_loader)

    out_yaml = {"args": kwargs, "test_metrics": test_out[0]}
    out_str = yaml.dump(out_yaml, indent=2, default_flow_style=False)

    with open(Path(save_dir) / "test_results.yaml", "w") as fp:
        fp.write(out_str)

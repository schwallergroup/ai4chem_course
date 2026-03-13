import torch
import copy
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class ForwardFFN(pl.LightningModule):

    def __init__(self,
                 hidden_size: int,
                 layers: int = 2,
                 dropout: float = 0.0,
                 learning_rate: float = 7e-4,
                 min_lr: float = 1e-5,
                 input_dim: int = 2048,
                 output_dim: int = 1,
                 **kwargs):
        """__init__.

        Args:
            hidden_size (int): hidden_size
            layers (int): layers
            dropout (float): dropout
            learning_rate (float): learning_rate
            min_lr (float): min_lr
            input_dim (int): input_dim
            output_dim (int): output_dim
            kwargs:
        """
        super().__init__()
        self.save_hyperparameters()
        self.hidden_size = hidden_size
        self.layers = layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.learning_rate = learning_rate
        self.min_lr = min_lr

        # Define network
        self.activation = nn.ReLU()

        # Use identity
        self.output_activation = nn.Identity()
        self.mlp = MLPBlocks(input_size=self.input_dim,
                             hidden_size=self.hidden_size,
                             dropout=self.dropout,
                             num_layers=self.layers)
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.output_dim),
            self.output_activation)
        self.loss_fn = self.mse_loss

    def mse_loss(self, pred, targ, **kwargs):
        """ mse_loss.

        Args:
            pred (torch.tensor): Predictions
            targ (torch.tensor): Targets
        """
        mse_loss = F.mse_loss(pred, targ)
        return {"loss": mse_loss}

    def forward(self, fps):
        """forward.
        """
        fps = fps.float()
        output = self.mlp(fps)
        output = self.output_layer(output)
        return output

    def training_step(self, batch, batch_idx):
        """training_step.

        Args:
            batch:
            batch_idx:
        """
        preds = self.forward(batch['fps'])
        loss_dict = self.loss_fn(preds, batch['targs'])
        self.log("train_loss", loss_dict.get("loss"))
        return loss_dict

    def validation_step(self, batch, batch_idx):
        """validation_step.

        Args:
            batch:
            batch_idx:
        """
        preds = self.forward(batch['fps'])
        loss_dict = self.loss_fn(preds, batch['targs'])
        self.log("val_loss", loss_dict.get("loss"))
        return loss_dict

    def test_step(self, batch, batch_idx):
        """test_step.

        Args:
            batch:
            batch_idx:
        """
        preds = self.forward(batch['fps'])
        loss_dict = self.loss_fn(preds, batch['targs'])
        self.log("test_loss", loss_dict.get("loss"))
        return loss_dict

    def configure_optimizers(self):
        """configure_optimizers.
        """
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.learning_rate,
                                     weight_decay=0.0)
        decay_rate = 0.8
        start_lr = self.learning_rate
        steps_to_decay = 1
        min_decay_rate = self.min_lr / start_lr
        lr_lambda = lambda epoch: (np.maximum(
            decay_rate**(epoch // steps_to_decay), min_decay_rate))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lr_lambda=lr_lambda)
        ret = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "frequency": 1,
                "interval": "epoch"
            }
        }
        return ret


class MLPBlocks(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float,
        num_layers: int,
    ):
        """__init__.

        Args:
            input_size (int): input_size
            hidden_size (int): hidden_size
            dropout (float): dropout
            num_layers (int): num_layers
        """
        super().__init__()
        self.activation = nn.ReLU()
        self.dropout_layer = nn.Dropout(p=dropout)
        self.input_layer = nn.Linear(input_size, hidden_size)
        middle_layer = nn.Linear(hidden_size, hidden_size)
        self.layers = get_clones(middle_layer, num_layers - 1)

    def forward(self, x):
        """forward.

        Args:
            x:
        """
        output = x
        output = self.input_layer(x)
        output = self.dropout_layer(output)
        output = self.activation(output)
        for layer_index, layer in enumerate(self.layers):
            output = layer(output)
            output = self.dropout_layer(output)
            output = self.activation(output)
        return output


def get_clones(module, N):
    """get_clones.

    Args:
        module:
        N:
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

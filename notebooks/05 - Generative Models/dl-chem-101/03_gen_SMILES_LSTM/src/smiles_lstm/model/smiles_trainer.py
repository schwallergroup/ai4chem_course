"""
SMILES LSTM trainer class based on the REINVENT implementation.
"""
import csv
import os
from os import path
from typing import Union, Tuple
import rdkit
import torch
from smiles_lstm.model.smiles_dataset import Dataset
from smiles_lstm.model.smiles_vocabulary import SMILESTokenizer
from smiles_lstm.model.smiles_lstm import SmilesLSTM
from smiles_lstm.utils.misc import draw_smiles, progress_bar, save_smiles
import smiles_lstm.utils.load as load

rdkit.rdBase.DisableLog("rdApp.error")


class SmilesTrainer():
    """
    Trains a SMILES-based generative model using the input SMILES.
    """
    def __init__(self, model: SmilesLSTM, input_smiles : Union[dict, str],
                 epochs : int=10, learning_rate : float=0.0001,
                 batch_size : int=250, shuffle : bool=True,
                 augment : int=0, output_model_path : str="./output/", start_epoch : int=0,
                 learning_rate_scheduler : str="StepLR", gamma : float=0.8,
                 eval_num_samples : int=64, eval_batch_size : int=64) -> None:
        """
        Args:
        ----
            model (SmilesLSTM)                : The SMILES LSTM generative model.
            input_smiles (dict or str)        : If `str`, contains the path to the directory
                                                containing the training, testing, and
                                                validation data ("train.smi", "test.smi",
                                                and "valid.smi"). If `dict`, contains the
                                                training, testing, and validation data,
                                                with "train", "test", and "valid" as keys.
            epochs (int, optional)            : Number of epochs to train for.
                                                Defaults to 10.
            learning_rate (float, optional)   : Initial learning rate. Defaults
                                                to 0.0001.
            batch_size (int, optional)        : Batch size to use for training.
                                                Defaults to 250.
            shuffle (bool, optional)          : Whether or not to shuffle the data
                                                for training. Defaults to True.
            augment (int, optional)           : If nonzero, indicates how many
                                                SMILES to use for data augmentation.
            output_model_path (str, optional) : Directory in which to save the
                                                results. Defaults to "./output/".
            start_epoch (int)                 : Epoch at which to start training.
            learning_rate_scheduler (str)     : Type of learning rate scheduler
                                                ("StepLR" or "CosineAnnealingLR").
            gamma (float)                     : Gamma value for the StepLR learning
                                                rate scheduler.
            eval_num_samples (int)            : Number of samples to use for evaluating
                                                the generative model.
            eval_batch_size (int)             : Batch size to use during evaluation.
        """
        # define the model
        self._model = model

        # define parameters
        self._batch_size        = batch_size
        self._learning_rate     = learning_rate
        self._epochs            = epochs
        self._start_epoch       = start_epoch
        self._output_model_path = output_model_path
        self._shuffle           = shuffle
        self._use_augmentation  = augment
        self._eval_num_samples  = eval_num_samples
        self._eval_batch_size   = eval_batch_size

        # define the data
        (self._train_dataloader,
         self._test_dataloader,
         self._valid_dataloader) = self._load_smiles(input_smiles=input_smiles)

        # define the optimizer and scheduler
        self._optimizer = torch.optim.Adam(params=self._model.network.parameters(),
                                           lr=self._learning_rate)

        if learning_rate_scheduler == "CosineAnnealingLR":
            # cosine annealing schedule
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=self._optimizer,
                T_max=self._epochs,
            )
        elif learning_rate_scheduler == "StepLR":
            # step learning rate annealing schedule
            self._scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=self._optimizer,
                step_size=1,
                gamma=gamma,
            )
        else:
            raise ValueError("Please enter a valid value for ´learning_rate_scheduler´.")

        # create a directory for the model if one does not exist already
        if not os.path.exists(self._output_model_path):
            os.makedirs(self._output_model_path)
      
        # define paths for the output files
        params_filename                = f"{self._output_model_path}SmilesTrainer_params.csv"
        self._training_output_filename = f"{self._output_model_path}SmilesTrainer_training.csv"

        # write the basic hyperparameters to a CSV file
        with open(params_filename, "w", encoding="utf-8") as params_file:
            params_writer = csv.writer(params_file)
            param_names  = ["model type", "batch size", "learning rate",
                            "epochs", "start_epoch", "use shuffle",
                            "use augmentation", "eval num samples",
                            "eval batch size", "learning rate scheduler"]
            param_values = ["SmilesTrainer", self._batch_size, self._learning_rate,
                            self._epochs, self._start_epoch, self._shuffle,
                            self._use_augmentation, self._eval_num_samples,
                            self._eval_batch_size, learning_rate_scheduler]
            params_writer.writerow(param_names)  # the header is the parameter names
            params_writer.writerow(param_values)

        # placeholders for the loss
        self._train_loss      = None
        self._valid_loss      = None
        self._best_valid_loss = None
        self._best_epoch      = None

        # create the output file
        with open(self._training_output_filename, "w", encoding="utf-8") as training_file:
            training_writer = csv.writer(training_file)
            header = ["epoch", "learning rate", "training loss",
                      "validation loss", "fraction valid"]
            training_writer.writerow(header)

    def run(self):
        """
        Train the model for the specified number of epochs.
        """
        # begin training
        for epoch in range(self._start_epoch, self._epochs):

            self._train_epoch(self._train_dataloader)
            self._valid_epoch(self._valid_dataloader)

            # sample smiles and draw these
            sampled_smiles, nlls = self._model.sample_smiles(num=self._eval_num_samples,
                                                             batch_size=self._eval_batch_size)
            fraction_valid       = draw_smiles(
                path=f"{self._output_model_path}sampled_epoch{epoch}.png",
                smiles_list=sampled_smiles
            )
            save_smiles(smiles=sampled_smiles,
                        output_filename=f"{self._output_model_path}sampled_step{epoch}.smi")

            # write progress to the output CSV
            learning_rate = self._optimizer.param_groups[0]["lr"]
            with open(self._training_output_filename, "a", encoding="utf-8") as training_file:
                training_writer = csv.writer(training_file)
                progress        = [epoch,
                                   learning_rate,
                                   self._train_loss.item(),
                                   self._valid_loss.item(),
                                   fraction_valid]
                training_writer.writerow(progress)

            self._save_checkpoint(epoch)

    def _save_checkpoint(self, epoch : int) -> None:
        """
        Save a checkpoint of the current model and also keep track of the best
        validation loss seen so far.

        Args:
            epoch (int) : Current training epoch.
        """
        # save a checkpoint if model is optimal
        if self._best_valid_loss is None:
            self._best_valid_loss = self._valid_loss
        elif self._valid_loss < self._best_valid_loss:
            self._best_valid_loss = self._valid_loss
            self._best_epoch = epoch

        self._save_current_model(epoch)

    def _train_epoch(self, train_dataloader : torch.utils.data.DataLoader):
        """
        Perform one training epoch.

        Args:
        ----
            train_dataloader (torch.utils.data.DataLoader) : Dataloader containing
                                                             training data.
        """
        loss_tensor = torch.zeros(len(train_dataloader))
        self._model.network.train()
        dataloader_progress_bar = progress_bar(iterable=train_dataloader,
                                               total=len(train_dataloader))
        for batch_idx, batch in enumerate(dataloader_progress_bar):
            input_vectors          = batch.long()
            loss                   = self._calculate_loss(input_vectors)
            loss_tensor[batch_idx] = loss

            self._model.network.zero_grad()  # clear gradient
            self._optimizer.zero_grad()      # clear gradient
            loss.backward()
            self._optimizer.step()

        # update the training loss
        self._train_loss = torch.mean(loss_tensor)

        # update the scheduler
        self._scheduler.step()

    def _valid_epoch(self, valid_dataloader : torch.utils.data.DataLoader):
        """
        Perform one validation epoch.

        Args:
        ----
            valid_dataloader (torch.utils.data.DataLoader) : Dataloader containing
                                                             validation data.
        """
        loss_tensor = torch.zeros(len(valid_dataloader))
        self._model.network.eval()
        dataloader_progress_bar = progress_bar(iterable=valid_dataloader,
                                               total=len(valid_dataloader))
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader_progress_bar):
                input_vectors          = batch.long()
                loss                   = self._calculate_loss(input_vectors)
                loss_tensor[batch_idx] = loss

        # update the validation loss
        self._valid_loss = torch.mean(loss_tensor)

    def _initialize_dataloader(self, smiles_list : list) -> torch.utils.data.DataLoader:
        """
        Create a dataloader object for a list of SMILES.

        Args:
        ----
            smiles_list (list) : List of SMILES.

        Returns:
        -------
            torch.utils.data.DataLoader : Dataloader object.
        """
        smiles_list_augmented = []
        if self._use_augmentation:
            for smiles in smiles_list:
                smiles_list_augmented += self._augment(smiles=smiles,
                                                       n_permutations=self._use_augmentation)
        smiles_list += smiles_list_augmented

        dataset = Dataset(smiles_list=smiles_list,
                          vocabulary=self._model.vocabulary,
                          tokenizer=SMILESTokenizer())
        if len(dataset) == 0:
            raise IOError(f"No valid entries are present in the "
                          f"supplied file: {path}")

        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self._batch_size,
                                                 shuffle=self._shuffle,
                                                 collate_fn=Dataset.collate_fn,
                                                 drop_last=True)
        return dataloader

    def _augment(self, smiles : str, n_permutations : int) -> list:
        """
        Enumerate a SMILES by creating `n_permutations` of that SMILES with a
        shuffled node order.

        Args:
        ----
            smiles (str)                   : SMILES string to enumerate.
            n_permutations (int, optional) : Number of shuffled copies of the
                                             SMILES string.

        Returns:
        -------
            list: Contains shuffled SMILES strings.
        """
        molecule = rdkit.Chem.MolFromSmiles(smiles)

        try:
            permutations = [rdkit.Chem.MolToSmiles(molecule,
                                                   canonical=False,
                                                   doRandom=True,
                                                   isomericSmiles=False)
                            for _ in range(n_permutations)]
        except RuntimeError:
            permutations = [smiles]

        return permutations


    def _calculate_loss(self, input_vectors : torch.Tensor) -> torch.Tensor:
        """
        Calculate the negative log likelihood for the specified input sequences.

        Args:
        ----
            input_vectors (torch.Tensor) : Batch of input SMILES sequences.

        Returns:
        -------
            torch.Tensor : Batch of negative log likelihoods.
        """
        log_p = self._model.likelihood(input_vectors)
        return log_p.mean()

    def _save_current_model(self, epoch : int) -> None:
        """
        Saves the current model as the 'last' model seen thus far.

        Args:
        ----
            epoch (int) : Current training epoch.
        """
        model_path = f"{self._output_model_path}model.{epoch}.pth"
        self._model.save_state(path=model_path)

    def _load_smiles(self, input_smiles : Union[dict, str]) -> \
                     Tuple[list, list, list]:
        """
        Loads the SMILES into a dataloader, for each the training, testing, and
        validation set.

        Args:
        ----
            input_smiles (dict or str) : If str, contains the path to the directory
                                         containing the training, testing, and
                                         validation data ("train.smi", "test.smi",
                                         and "valid.smi"). If dict, contains the
                                         training, testing, and validation data,
                                         with "train", "test", and "valid" as keys.

        Returns:
        -------
            train_dataloader : Training set dataloader.
            test_dataloader  : Testing set dataloader.
            valid_dataloader : Validation set dataloader.
        """
        # load the SMILES into lists
        if isinstance(input_smiles, str):
            # load from file
            train_smiles = load.smiles(path=f"{input_smiles}train.smi")
            test_smiles  = load.smiles(path=f"{input_smiles}test.smi")
            valid_smiles = load.smiles(path=f"{input_smiles}valid.smi")
        elif isinstance(input_smiles, dict):
            # get values from dictionary
            train_smiles = input_smiles["train"]
            test_smiles  = input_smiles["test"]
            valid_smiles = input_smiles["valid"]
        else:
            raise NotImplementedError

        # create the dataloader from the SMILES lists
        train_dataloader = self._initialize_dataloader(smiles_list=train_smiles)
        test_dataloader  = self._initialize_dataloader(smiles_list=test_smiles)
        valid_dataloader = self._initialize_dataloader(smiles_list=valid_smiles)

        return train_dataloader, test_dataloader, valid_dataloader

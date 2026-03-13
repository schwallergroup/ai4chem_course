"""
Implementation of a SMILES dataset.
"""
from typing import Tuple, Union
import torch
from smiles_lstm.model.smiles_vocabulary import SMILESTokenizer, Vocabulary


class Dataset(torch.utils.data.Dataset):
    """
    Custom PyTorch Dataset that takes a file containing \n separated SMILES.
    """

    def __init__(self, smiles_list : list, vocabulary : Vocabulary,
                 tokenizer : SMILESTokenizer) -> None:
        self._vocabulary  = vocabulary
        self._tokenizer   = tokenizer
        self._smiles_list = list(smiles_list)

    def __getitem__(self, i : int) -> torch.Tensor:
        smi     = self._smiles_list[i]
        tokens  = self._tokenizer.tokenize(smi)
        encoded = self._vocabulary.encode(tokens)
        return torch.tensor(encoded, dtype=torch.long)  # pylint: disable=E1102

    def __len__(self) -> int:
        return len(self._smiles_list)

    @staticmethod
    def collate_fn(encoded_seqs : list) -> torch.Tensor:
        """
        Converts a list of encoded sequences into a padded tensor.
        """
        max_length   = max([seq.size(0) for seq in encoded_seqs])
        collated_arr = torch.zeros(len(encoded_seqs),
                                   max_length,
                                   dtype=torch.long)  # padded with zeros
        for i, seq in enumerate(encoded_seqs):
            collated_arr[i, :seq.size(0)] = seq
        return collated_arr


def calculate_nlls_from_model(model : torch.nn.Module, smiles : Union[list, iter],
                              batch_size : int=128) -> Tuple[iter, int]:
    """
    Calculates NLL for a set of SMILES strings.

    Params:
    ------
        model (torch.nn.Module) : Model object.
        smiles (list or iter)   : List or iterator with all SMILES strings.
        batch_size (int)        : Batch size to use for this computation.

    Returns:
    -------
        iterator : Contains the nlls of the items in the dataloader.
        int      : the length of the dataloader.
    """
    dataset     = Dataset(smiles, model.vocabulary, model.tokenizer)
    _dataloader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              collate_fn=Dataset.collate_fn)

    def _iterator(dataloader):
        for batch in dataloader:
            nlls = model.likelihood(batch.long())
            yield nlls.data.cpu().numpy()

    return _iterator(_dataloader), len(_dataloader)

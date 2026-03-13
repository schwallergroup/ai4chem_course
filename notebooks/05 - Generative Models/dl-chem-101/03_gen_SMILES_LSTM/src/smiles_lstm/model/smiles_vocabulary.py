"""
Vocabulary helper class based on the REINVENT implementation.
"""
import re
from typing import Union
import rdkit
import numpy as np


class Vocabulary:
    """
    Stores the tokens and their conversion to vocabulary indexes.
    """
    def __init__(self, tokens : Union[dict, None]=None, starting_id : int=0) -> None:
        self._tokens = {}
        self._current_id = starting_id

        if tokens:
            for token, idx in tokens.items():
                self._add(token, idx)
                self._current_id = max(self._current_id, idx + 1)

    def __getitem__(self, token_or_id : str) -> int:
        return self._tokens[token_or_id]

    def add(self, token : str) -> int:
        """
        Adds a token to the vocabulary.
        """
        if not isinstance(token, str):
            raise TypeError("Token is not a string")
        if token in self:
            return self[token]
        self._add(token, self._current_id)
        self._current_id += 1
        return self._current_id - 1

    def update(self, tokens : list) -> list:
        """
        Adds many tokens.
        """
        return [self.add(token) for token in tokens]

    def __delitem__(self, token_or_id : Union[str, int]) -> None:
        other_val = self._tokens[token_or_id]
        del self._tokens[other_val]
        del self._tokens[token_or_id]

    def __contains__(self, token_or_id : Union[str, int]) -> None:
        return token_or_id in self._tokens

    def __eq__(self, other_vocabulary : "Vocabulary") -> int:
        return self._tokens == other_vocabulary._tokens  # pylint: disable=W0212

    def __len__(self) -> int:
        return len(self._tokens) // 2

    def encode(self, tokens : list) -> np.ndarray:
        """
        Encodes a list of tokens as vocabulary indexes.
        """
        vocab_index = np.zeros(len(tokens), dtype=np.float32)
        for i, token in enumerate(tokens):
            vocab_index[i] = self._tokens[token]
        return vocab_index

    def decode(self, vocab_index : np.ndarray) -> list:
        """
        Decodes a vocabulary index matrix to a list of tokens.
        """
        tokens = []
        for idx in vocab_index:
            tokens.append(self[idx])
        return tokens

    def _add(self, token : str, idx : int) -> None:
        if idx not in self._tokens:
            self._tokens[token] = idx
            self._tokens[idx] = token
        else:
            raise ValueError("IDX already present in vocabulary")

    def tokens(self) -> list:
        """
        Returns the tokens from the vocabulary.
        """
        return [t for t in self._tokens if isinstance(t, str)]


class SMILESTokenizer:
    """
    Handles the tokenization and untokenization of SMILES.
    """

    REGEXPS = {
        "brackets": re.compile(r"(\[[^\]]*\])"),
        "2_ring_nums": re.compile(r"(%\d{2})"),
        "brcl": re.compile(r"(Br|Cl)")
    }
    REGEXP_ORDER = ["brackets", "2_ring_nums", "brcl"]

    def tokenize(self, data : str, with_begin_and_end : bool=True) -> list:
        """
        Tokenizes a SMILES string.
        """
        def split_by(data, regexps):
            if not regexps:
                return list(data)
            regexp = self.REGEXPS[regexps[0]]
            splitted = regexp.split(data)
            tokens = []
            for i, split in enumerate(splitted):
                if i % 2 == 0:
                    tokens += split_by(split, regexps[1:])
                else:
                    tokens.append(split)
            return tokens

        tokens = split_by(data, self.REGEXP_ORDER)
        if with_begin_and_end:
            tokens = ["^"] + tokens + ["$"]
        return tokens

    def untokenize(self, tokens : list) -> str:
        """
        Untokenizes a SMILES string.
        """
        smi = ""
        for token in tokens:
            if token == "$":
                break
            if token != "^":
                smi += token
        return smi


def create_vocabulary(smiles_list : list, tokenizer : SMILESTokenizer,
                      canonical : bool=True) -> Vocabulary:
    """
    Creates a vocabulary for the SMILES syntax.
    """
    if not canonical:
        noncanon_smiles_list = []
        for smiles in smiles_list:
            molecule = rdkit.Chem.MolFromSmiles(smiles)

            try:
                noncanon_smiles_list.append(
                    rdkit.Chem.MolToSmiles(molecule,
                                           canonical=False,
                                           doRandom=True,
                                           isomericSmiles=False)
                )
            except:
                pass

        smiles_list += noncanon_smiles_list

    tokens = set()
    for smi in smiles_list:
        tokens.update(tokenizer.tokenize(smi, with_begin_and_end=False))

    vocabulary = Vocabulary()
    vocabulary.update(["$", "^"] + sorted(tokens))  # end token is 0 (also counts as padding)
    return vocabulary

import torch
from smiles_lstm.model.smiles_vocabulary import SMILESTokenizer
from smiles_lstm.model.smiles_lstm import SmilesLSTM

def load_from_file(file_path : str, sampling_mode : bool=False):
    """
    Loads a model from a single file.
    Params:
    ------
        file_path (str) : Input file path.
    Returns:
    -------
        SmilesLSTM : New instance of the RNN, or an exception if it was not
                     possible to load it.
    """
    model = torch.load(file_path, map_location='cpu')
    if sampling_mode:
        model.network.eval()

    return model

# batch size hard-coded as 1 here just for illustration --> this method samples 1 SMILES
def sample(batch_size=1, model=None):
    loss = torch.nn.NLLLoss(reduction='none')
    start_token = torch.zeros(batch_size,
                              dtype=torch.long,
                              device='cpu')
    start_token[:] = model.vocabulary["^"]
    input_vector = start_token
    sequences = [
        model.vocabulary["^"] * torch.ones([batch_size, 1],
                                                   dtype=torch.long,
                                                   device='cpu')
    ]
    # NOTE: The first token never gets added in the loop so
    # the sequences are initialized with a start token
    hidden_state = None
    nlls = torch.zeros(batch_size)
    for _ in range(256 - 1):
        logits, hidden_state = model.network(input_vector.unsqueeze(1), hidden_state)
        # logits are the output from the RNN
        logits        = logits.squeeze(1)
        # we apply the "Softmax" function on the logits to obtain the probabilities
        probabilities = logits.softmax(dim=1)
        log_probs     = logits.log_softmax(dim=1)
        input_vector  = torch.multinomial(probabilities, 1).view(-1)
        sequences.append(input_vector.view(-1, 1))
        nlls += loss(log_probs, input_vector)
        if input_vector.sum() == 0:
            break

    sequences = torch.cat(sequences, 1)
    return sequences.data, nlls

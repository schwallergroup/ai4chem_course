import torch


def load_from_file(file_path, device):
        """
        Loads a model from a single file.

        Params:
        ------
            file_path (str) : Input file path.
            device (str) : Device the model will be loaded onto.
            sampling_mode (bool) : Whether to set the model to eval mode.

        Returns:
        -------
            SmilesLSTM : New instance of the RNN, or an exception if it was not
                         possible to load it.
        """
        # Load model directly to target device
        model = torch.load(file_path, weights_only=False, map_location=device)
        model._device = device

        return model

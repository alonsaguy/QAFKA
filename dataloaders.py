import torch
from torch.utils.data import TensorDataset

def CreateDataLoader(X, y, batch_size):
    """
        Creates data loader for X, y pairs
        :param X: Tensor [# of observations, numOfBins] for the observations
        :param y: Tensor [# of observations] for the predictions
        :param batch_size: Int for batch size
        :return: data loader for training/ testing routines
    """
    data_loader = torch.utils.data.DataLoader(TensorDataset(X, y), batch_size=batch_size)
    return data_loader

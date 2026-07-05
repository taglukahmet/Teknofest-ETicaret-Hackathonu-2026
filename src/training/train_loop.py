import torch.nn as nn
from torch.utils.data import DataLoader

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> None:
    """
    (Tasks: Member 3 & 5)
    Executes the forward/backward passes. 
    Must contain W&B tracking hooks for logging loss and validation metrics.
    """
    pass
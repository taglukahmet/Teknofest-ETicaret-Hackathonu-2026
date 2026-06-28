import polars as pl
import torch.nn as nn
from torch.utils.data import DataLoader

def run_test_inference(model: nn.Module, test_loader: DataLoader, device: str) -> pl.DataFrame:
    """
    (Task: You)
    Passes the tokenized submission pairs through the trained model without tracking gradients.
    Extracts the raw logits, applies softmax/sigmoid to get probabilities.
    Returns a Polars DataFrame with columns: ['id', 'probability']
    """
    # Boilerplate return to prevent IDE errors
    return pl.DataFrame()
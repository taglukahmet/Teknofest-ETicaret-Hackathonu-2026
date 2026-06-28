from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

def create_dataloaders(parquet_path: str, tokenizer: PreTrainedTokenizerFast, batch_size: int) -> tuple[DataLoader, DataLoader]:
    """
    (Task: Member 3)
    Streams data from disk using Hugging Face datasets.
    Returns the mapped (train_loader, val_loader).
    """
    return None, None
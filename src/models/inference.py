import torch
import torch.nn as nn
import polars as pl
from torch.utils.data import DataLoader
from tqdm import tqdm

def run_test_inference(model: nn.Module, test_loader: DataLoader, device: str) -> pl.DataFrame:
    """
    Executes a high-throughput, memory-safe forward pass over the blind Kaggle test set.
    Extracts raw logits, normalizes them via Softmax, and structures the output.
    
    Args:
        model: The trained Cross-Encoder architecture.
        test_loader: DataLoader yielding batches of test data (must include 'id', 'input_ids', 'attention_mask').
        device: The target hardware accelerator.
        
    Returns:
        pl.DataFrame: A Polars dataframe with columns ['id', 'probability']
    """
    # 1. Lock the model into deterministic evaluation mode
    model.eval()

    all_ids = []
    all_probabilities = []

    # Initialize the progress bar for the terminal
    progress_bar = tqdm(test_loader, desc="Running Inference", leave=True)

    # 2. Disable the Autograd engine to prevent VRAM overflow
    with torch.no_grad():
        for batch in progress_bar:
            # Map the tensors to the target frame
            ids = batch['id']
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # 3. Execute the forward pass 
            logits = model(input_ids=input_ids, attention_mask=attention_mask)

            # 4. Normalize logits to probabilities
            # Dim=1 ensures we calculate Softmax across the 2 classes for each row in the batch
            probabilities = torch.softmax(logits, dim=1)

            # We strictly want the probability of Class 1 (VAR)
            class_1_probs = probabilities[:,1].cpu().numpy().tolist()

            # Append to our tracking lists
            all_ids.extend(ids)
            all_probabilities.extend(class_1_probs)

    # 5. Construct the final analytical structure
    # Polars is used here to ensure strict type safety (strings for IDs, floats for probs)
    submission_df = pl.DataFrame({
        "id": all_ids,
        "probability": all_probabilities
    })

    return submission_df
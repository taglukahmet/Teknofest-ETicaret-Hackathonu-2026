from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast, DataCollatorWithPadding

from src.config import config

# Contract with the data pipeline (src/data/*): the parquet at parquet_path must
# carry these columns after format_migration -> relational_merge -> negative_sample
# -> text_serialize -> kfold_split has run.
TEXT_COLUMN = "input_text"
LABEL_COLUMN = "label"
FOLD_COLUMN = "fold"


def _tokenize(batch, tokenizer: PreTrainedTokenizerFast):
    return tokenizer(
        batch[TEXT_COLUMN],
        truncation=True,
        max_length=config.MAX_SEQUENCE_LENGTH,
    )


def _prepare_split(dataset: Dataset, tokenizer: PreTrainedTokenizerFast) -> Dataset:
    dataset = dataset.map(lambda batch: _tokenize(batch, tokenizer), batched=True)
    keep = {"input_ids", "attention_mask", LABEL_COLUMN}
    dataset = dataset.remove_columns([c for c in dataset.column_names if c not in keep])
    dataset = dataset.rename_column(LABEL_COLUMN, "labels")
    dataset.set_format(type="torch")
    return dataset


def create_dataloaders(
    parquet_path: str,
    tokenizer: PreTrainedTokenizerFast,
    batch_size: int,
    val_fold: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """
    (Task: Member 3)
    Streams data from disk using Hugging Face datasets.
    Returns the mapped (train_loader, val_loader).
    """
    # Dataset.from_parquet backs the table with a memory-mapped Arrow file,
    # so this does not load the full (potentially multi-GB) frame into RAM.
    full = Dataset.from_parquet(parquet_path)

    train_raw = full.filter(lambda row: row[FOLD_COLUMN] != val_fold)
    val_raw = full.filter(lambda row: row[FOLD_COLUMN] == val_fold)

    train_dataset = _prepare_split(train_raw, tokenizer)
    val_dataset = _prepare_split(val_raw, tokenizer)

    # Dynamic padding per-batch (vs. always padding to MAX_SEQUENCE_LENGTH) keeps
    # training throughput up on the 6GB laptop GPUs the team is training on.
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # num_workers=0: Windows spawns worker processes rather than forking, which
    # re-imports this module per worker; keeping it single-process avoids that cost
    # for a dataset that's already memory-mapped (no need for background prefetch).
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader

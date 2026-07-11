from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast, DataCollatorWithPadding

from src.config import config

# Contract with the data pipeline (src/data/*): the parquet at parquet_path must
# carry these columns after format_migration -> relational_merge -> negative_sample
# -> text_serialize -> kfold_split has run.
ID_COLUMN = "id"
TEXT_COLUMN = "input_text"
LABEL_COLUMN = "label"
FOLD_COLUMN = "fold"


def _tokenize(texts: list[str], tokenizer: PreTrainedTokenizerFast):
    return tokenizer(
        texts,
        truncation=True,
        max_length=config.MAX_SEQUENCE_LENGTH,
    )


def _prepare_split(dataset: Dataset, tokenizer: PreTrainedTokenizerFast) -> Dataset:
    # input_columns=[TEXT_COLUMN]: only that column is materialized per row instead
    # of the full record, same reasoning as the filter() optimization below.
    dataset = dataset.map(
        lambda texts: _tokenize(texts, tokenizer),
        batched=True,
        input_columns=[TEXT_COLUMN],
    )
    keep = {ID_COLUMN, "input_ids", "attention_mask", LABEL_COLUMN}
    dataset = dataset.remove_columns([c for c in dataset.column_names if c not in keep])
    dataset = dataset.rename_column(LABEL_COLUMN, "labels")
    # id stays a plain string, so it's excluded from the torch-formatted columns and
    # passed through as-is via output_all_columns (ensemble.py joins OOF/test
    # predictions back to the pair table on this id).
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"], output_all_columns=True)
    return dataset


def _make_collate_fn(tokenizer: PreTrainedTokenizerFast):
    pad_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def collate(features):
        # HF's batched __getitems__ hands us a dict-of-lists (columnar); DataLoader
        # falls back to a list-of-dicts if that fast path isn't available. Either
        # way, "id" (a string) has to be pulled out before the tokenizer pads the
        # rest, since it can't be turned into a tensor.
        if isinstance(features, dict):
            ids = list(features.pop(ID_COLUMN))
            batch = pad_collator(features)
        else:
            ids = [f.pop(ID_COLUMN) for f in features]
            batch = pad_collator(features)
        batch[ID_COLUMN] = ids
        return batch

    return collate


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

    # input_columns=[FOLD_COLUMN]: the filter predicate only needs the fold value,
    # so only that column gets deserialized per row instead of the whole record.
    train_raw = full.filter(lambda fold: fold != val_fold, input_columns=[FOLD_COLUMN])
    val_raw = full.filter(lambda fold: fold == val_fold, input_columns=[FOLD_COLUMN])

    train_dataset = _prepare_split(train_raw, tokenizer)
    val_dataset = _prepare_split(val_raw, tokenizer)

    # Dynamic padding per-batch (vs. always padding to MAX_SEQUENCE_LENGTH) keeps
    # training throughput up on the 6GB laptop GPUs the team is training on.
    collate_fn = _make_collate_fn(tokenizer)

    # num_workers=0: Windows spawns worker processes rather than forking, which
    # re-imports this module per worker; keeping it single-process avoids that cost
    # for a dataset that's already memory-mapped (no need for background prefetch).
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader

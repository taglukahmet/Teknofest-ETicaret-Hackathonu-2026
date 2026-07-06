import polars as pl
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
# Config entegrasyonu
import config

def apply_group_kfold(df: pl.DataFrame, n_splits: int) -> pl.DataFrame:
    """
    (Task: Member 2)
    Applies StratifiedGroupKFold on the 'term_id' column using config.RANDOM_SEED.
    Ensures fold tracking is integer based and stable.
    """
    if df.is_empty():
        return df.with_columns(pl.lit(None, dtype=pl.Int32).alias("fold"))

    X_dummy = np.zeros(len(df))
    y = df["label"].to_numpy()
    groups = df["term_id"].to_numpy()
    
    fold_assignments = np.full(len(df), -1, dtype=np.int32)
    
    # random_state değerini merkezi config dosyasından çekiyoruz
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_SEED)
    
    for fold_idx, (_, val_indices) in enumerate(sgkf.split(X_dummy, y, groups=groups)):
        fold_assignments[val_indices] = fold_idx
        
    df_with_folds = df.with_columns(
        pl.Series(fold_assignments, dtype=pl.Int32).alias("fold")
    )
    
    return df_with_folds
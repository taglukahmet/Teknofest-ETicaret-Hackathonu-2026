import polars as pl

def apply_group_kfold(df: pl.DataFrame, n_splits: int) -> pl.DataFrame:
    """
    (Task: Member 2)
    Note: Requires eager evaluation (DataFrame, not LazyFrame) for scikit-learn compatibility.
    Applies GroupKFold on the 'term_id' column to prevent data leakage.
    Returns dataframe with a new 'fold' integer column.
    """
    return df
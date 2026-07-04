from pathlib import Path

import polars as pl


DEFAULT_PROBABILITY_COLUMNS = (
    "probability",
    "class_1_probability",
    "positive_probability",
    "score",
    "prediction",
)


def _read_prediction_file(path: str | Path) -> pl.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    if path.suffix.lower() == ".csv":
        return pl.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pl.read_parquet(path)

    raise ValueError(f"Unsupported prediction file type: {path.suffix}")


def _resolve_probability_column(
    df: pl.DataFrame,
    requested_column: str | None,
) -> str:
    if requested_column is not None:
        if requested_column not in df.columns:
            raise ValueError(f"Probability column not found: {requested_column}")
        return requested_column

    for column in DEFAULT_PROBABILITY_COLUMNS:
        if column in df.columns:
            return column

    raise ValueError(
        "Could not find a probability column. Expected one of: "
        + ", ".join(DEFAULT_PROBABILITY_COLUMNS)
    )


def _prepare_prediction_frame(
    path: str | Path,
    model_index: int,
    id_column: str,
    probability_column: str | None,
) -> tuple[pl.DataFrame, str]:
    df = _read_prediction_file(path)

    if id_column not in df.columns:
        raise ValueError(f"ID column not found in {path}: {id_column}")

    resolved_probability_column = _resolve_probability_column(df, probability_column)
    model_probability_column = f"model_{model_index}_probability"

    prepared = df.select(
        pl.col(id_column).cast(pl.Utf8).alias(id_column),
        pl.col(resolved_probability_column).cast(pl.Float64).alias(model_probability_column),
    )

    has_duplicate_ids = prepared.select(pl.col(id_column).is_duplicated().any()).item()
    if has_duplicate_ids:
        raise ValueError(f"Duplicate IDs found in prediction file: {path}")

    return prepared, model_probability_column


def soft_vote_predictions(
    prediction_files: list[str],
    id_column: str = "id",
    probability_column: str | None = None,
    output_column: str = "blended_probability",
) -> pl.DataFrame:
    """
    (Task: Member 5)
    Loads multiple prediction DataFrames (either OOF validation or blind test outputs).
    Averages the raw probability scores across all models to stabilize the final prediction.
    Returns a single Polars DataFrame containing the 'id' and the 'blended_probability'.
    """
    if not prediction_files:
        raise ValueError("prediction_files cannot be empty.")

    prepared_frames = []
    probability_columns = []

    for model_index, path in enumerate(prediction_files, start=1):
        frame, model_probability_column = _prepare_prediction_frame(
            path=path,
            model_index=model_index,
            id_column=id_column,
            probability_column=probability_column,
        )
        prepared_frames.append(frame)
        probability_columns.append(model_probability_column)

    blended = prepared_frames[0]
    expected_rows = blended.height

    for frame in prepared_frames[1:]:
        blended = blended.join(frame, on=id_column, how="inner")
        if blended.height != expected_rows:
            raise ValueError(
                "Prediction files do not contain the same ID set. "
                "Align by id before averaging; never average raw arrays by row order."
            )

    return blended.with_columns(
        pl.mean_horizontal(probability_columns).alias(output_column)
    ).select(id_column, output_column)


def build_submission_frame(
    blended_predictions: pl.DataFrame,
    threshold: float,
    id_column: str = "id",
    probability_column: str = "blended_probability",
    target_column: str = "label",
) -> pl.DataFrame:
    """
    Applies the chosen threshold and returns a Kaggle-style id + label dataframe.
    """
    if id_column not in blended_predictions.columns:
        raise ValueError(f"ID column not found: {id_column}")
    if probability_column not in blended_predictions.columns:
        raise ValueError(f"Probability column not found: {probability_column}")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    return blended_predictions.select(
        pl.col(id_column),
        (pl.col(probability_column) >= threshold).cast(pl.Int8).alias(target_column),
    )

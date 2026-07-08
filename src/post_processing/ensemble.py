from pathlib import Path

import polars as pl


def _read_prediction_file(path: str | Path) -> pl.LazyFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    if path.suffix.lower() == ".csv":
        return pl.scan_csv(path)
    if path.suffix.lower() == ".parquet":
        return pl.scan_parquet(path)

    raise ValueError(f"Unsupported prediction file type: {path.suffix}")


def soft_vote_predictions(
    prediction_files: list[str | Path],
    id_column: str = "id",
    probability_column: str = "probability",
    output_column: str = "blended_probability",
) -> pl.LazyFrame:
    """
    (Task: Member 5)
    Loads multiple prediction DataFrames (either OOF validation or blind test outputs).
    Averages the raw probability scores across all models to stabilize the final prediction.
    Returns a lazy Polars frame containing the 'id' and the 'blended_probability'.
    """
    if not prediction_files:
        raise ValueError("prediction_files cannot be empty.")

    blended: pl.LazyFrame | None = None
    probability_columns = []

    for model_index, path in enumerate(prediction_files, start=1):
        df = _read_prediction_file(path)
        columns = df.collect_schema().names()
        missing_columns = [
            column for column in (id_column, probability_column) if column not in columns
        ]
        if missing_columns:
            raise ValueError(f"{path} is missing columns: {', '.join(missing_columns)}")

        model_probability_column = f"model_{model_index}_probability"
        frame = df.select(
            pl.col(id_column).cast(pl.Utf8).alias(id_column),
            pl.col(probability_column).cast(pl.Float64).alias(model_probability_column),
        )

        if blended is None:
            blended = frame
        else:
            blended = blended.join(frame, on=id_column, how="inner", validate="1:1")

        probability_columns.append(model_probability_column)

    assert blended is not None
    return blended.with_columns(
        pl.mean_horizontal(probability_columns).alias(output_column)
    ).select(id_column, output_column)


def build_submission_frame(
    blended_predictions: pl.DataFrame | pl.LazyFrame,
    threshold: float,
    id_column: str = "id",
    probability_column: str = "blended_probability",
    target_column: str = "label",
) -> pl.LazyFrame:
    """
    Applies the chosen threshold and returns a Kaggle-style id + label dataframe.
    """
    frame = (
        blended_predictions.lazy()
        if isinstance(blended_predictions, pl.DataFrame)
        else blended_predictions
    )
    columns = frame.collect_schema().names()

    if id_column not in columns:
        raise ValueError(f"ID column not found: {id_column}")
    if probability_column not in columns:
        raise ValueError(f"Probability column not found: {probability_column}")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    return frame.select(
        pl.col(id_column),
        (pl.col(probability_column) >= threshold).cast(pl.Int8).alias(target_column),
    )


def _write_smoke_prediction_files(output_dir: Path) -> list[str]:
    ids = ["PAIR_001", "PAIR_002", "PAIR_003", "PAIR_004", "PAIR_005", "PAIR_006"]
    model_probabilities = [
        [0.10, 0.30, 0.61, 0.72, 0.42, 0.88],
        [0.12, 0.28, 0.58, 0.75, 0.39, 0.91],
        [0.08, 0.34, 0.66, 0.69, 0.46, 0.84],
        [0.15, 0.25, 0.63, 0.78, 0.44, 0.86],
        [0.11, 0.31, 0.60, 0.73, 0.41, 0.89],
    ]

    prediction_files = []
    for model_index, probabilities in enumerate(model_probabilities, start=1):
        path = output_dir / f"model_{model_index}_oof.csv"
        pl.DataFrame({"id": ids, "probability": probabilities}).write_csv(path)
        prediction_files.append(str(path))

    return prediction_files


def _run_smoke_test() -> None:
    import sys
    import tempfile

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.post_processing.metrics import find_optimal_threshold, macro_f1_for_threshold

    labels = pl.DataFrame(
        {
            "id": ["PAIR_001", "PAIR_002", "PAIR_003", "PAIR_004", "PAIR_005", "PAIR_006"],
            "label": [0, 0, 1, 1, 0, 1],
        }
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        prediction_files = _write_smoke_prediction_files(Path(tmp_dir))
        blended = soft_vote_predictions(prediction_files).collect()
        scored = blended.join(labels, on="id", how="inner")

        threshold = find_optimal_threshold(
            scored["blended_probability"].to_numpy(),
            scored["label"].to_numpy(),
        )
        macro_f1 = macro_f1_for_threshold(
            scored["blended_probability"].to_numpy(),
            scored["label"].to_numpy(),
            threshold,
        )
        submission = build_submission_frame(blended, threshold).collect()

        print("Blended probabilities")
        print(blended.write_csv())
        print()
        print(f"Best threshold: {threshold}")
        print(f"Validation Macro-F1: {macro_f1:.4f}")
        print()
        print("Submission preview")
        print(submission.write_csv())


if __name__ == "__main__":
    _run_smoke_test()

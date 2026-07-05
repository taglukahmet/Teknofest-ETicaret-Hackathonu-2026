from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import polars as pl

from src.config import config
from src.mlops import finish_wandb_run, init_wandb_run, log_wandb_metrics
from src.post_processing.ensemble import build_submission_frame, soft_vote_predictions
from src.post_processing.metrics import find_optimal_threshold, macro_f1_for_threshold


@dataclass(frozen=True)
class Member5PipelineResult:
    threshold: float
    validation_macro_f1: float
    oof_blended_path: str
    oof_scored_path: str
    metrics_path: str
    submission_path: str | None = None


def _read_table(path: str | Path) -> pl.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() == ".csv":
        return pl.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pl.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _write_table(df: pl.DataFrame, path: Path) -> None:
    if path.suffix.lower() == ".csv":
        df.write_csv(path)
        return
    if path.suffix.lower() == ".parquet":
        df.write_parquet(path)
        return
    raise ValueError(f"Unsupported output file type: {path.suffix}")


def _load_labels(labels_file: str | Path, id_column: str, label_column: str) -> pl.DataFrame:
    labels = _read_table(labels_file)
    missing_columns = [
        column for column in (id_column, label_column) if column not in labels.columns
    ]
    if missing_columns:
        raise ValueError(f"Label file is missing columns: {', '.join(missing_columns)}")

    prepared = labels.select(
        pl.col(id_column).cast(pl.Utf8).alias(id_column),
        pl.col(label_column).cast(pl.Int8).alias(label_column),
    )

    if prepared.select(pl.col(id_column).is_duplicated().any()).item():
        raise ValueError(f"Duplicate IDs found in label file: {labels_file}")
    if not set(prepared[label_column].unique().to_list()).issubset({0, 1}):
        raise ValueError(f"{label_column} must contain only 0 and 1 values.")

    return prepared


def run_member5_pipeline(
    oof_prediction_files: list[str | Path],
    oof_labels_file: str | Path,
    output_dir: str | Path = config.OUTPUT_DIR,
    test_prediction_files: list[str | Path] | None = None,
    id_column: str = "id",
    label_column: str = "label",
    probability_column: str | None = None,
    blended_column: str = "blended_probability",
    min_threshold: float = 0.10,
    max_threshold: float = 0.90,
    step: float = 0.01,
    submission_filename: str = "submission.csv",
    use_wandb: bool = False,
    wandb_project: str = "teknofest-eticaret-2026",
    wandb_run_name: str | None = None,
    wandb_mode: str | None = None,
) -> Member5PipelineResult:
    """
    Runs the Member 5 post-processing workflow on real OOF and optional test files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    oof_blended = soft_vote_predictions(
        [str(path) for path in oof_prediction_files],
        id_column=id_column,
        probability_column=probability_column,
        output_column=blended_column,
    )
    labels = _load_labels(oof_labels_file, id_column=id_column, label_column=label_column)
    oof_scored = oof_blended.join(labels, on=id_column, how="inner")

    if oof_scored.height != oof_blended.height:
        raise ValueError(
            "OOF labels do not cover the same ID set as the blended predictions. "
            "Check that validation predictions and labels use the same IDs."
        )

    threshold = find_optimal_threshold(
        oof_scored[blended_column].to_numpy(),
        oof_scored[label_column].to_numpy(),
        min_threshold=min_threshold,
        max_threshold=max_threshold,
        step=step,
    )
    validation_macro_f1 = macro_f1_for_threshold(
        oof_scored[blended_column].to_numpy(),
        oof_scored[label_column].to_numpy(),
        threshold=threshold,
    )

    oof_blended_path = output_path / "oof_blended.csv"
    oof_scored_path = output_path / "oof_scored.csv"
    metrics_path = output_path / "member5_metrics.json"
    _write_table(oof_blended, oof_blended_path)
    _write_table(oof_scored, oof_scored_path)

    submission_path: Path | None = None
    if test_prediction_files:
        test_blended = soft_vote_predictions(
            [str(path) for path in test_prediction_files],
            id_column=id_column,
            probability_column=probability_column,
            output_column=blended_column,
        )
        submission = build_submission_frame(
            blended_predictions=test_blended,
            threshold=threshold,
            id_column=id_column,
            probability_column=blended_column,
            target_column=label_column,
        )
        submission_path = output_path / submission_filename
        _write_table(submission, submission_path)

    result = Member5PipelineResult(
        threshold=threshold,
        validation_macro_f1=validation_macro_f1,
        oof_blended_path=str(oof_blended_path),
        oof_scored_path=str(oof_scored_path),
        metrics_path=str(metrics_path),
        submission_path=str(submission_path) if submission_path else None,
    )

    metrics: dict[str, Any] = {
        **asdict(result),
        "oof_model_count": len(oof_prediction_files),
        "test_model_count": len(test_prediction_files or []),
        "oof_row_count": oof_scored.height,
        "min_threshold": min_threshold,
        "max_threshold": max_threshold,
        "threshold_step": step,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    run = init_wandb_run(
        project=wandb_project,
        run_name=wandb_run_name,
        mode=wandb_mode,
        enabled=use_wandb,
        config_values={
            "member": "Member 5",
            "oof_model_count": len(oof_prediction_files),
            "test_model_count": len(test_prediction_files or []),
            "min_threshold": min_threshold,
            "max_threshold": max_threshold,
            "threshold_step": step,
        },
        tags=["member5", "post-processing"],
    )
    try:
        log_wandb_metrics(
            run,
            {
                "threshold": threshold,
                "validation_macro_f1": validation_macro_f1,
                "oof_row_count": oof_scored.height,
            },
            prefix="post_processing",
        )
    finally:
        finish_wandb_run(run)

    return result

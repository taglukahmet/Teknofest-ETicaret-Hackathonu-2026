import tempfile
import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.post_processing import (
    build_submission_frame,
    find_optimal_threshold,
    macro_f1_for_threshold,
    soft_vote_predictions,
)


def _write_fake_prediction_files(output_dir: Path) -> list[str]:
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


def main() -> None:
    labels = pl.DataFrame(
        {
            "id": ["PAIR_001", "PAIR_002", "PAIR_003", "PAIR_004", "PAIR_005", "PAIR_006"],
            "label": [0, 0, 1, 1, 0, 1],
        }
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        prediction_files = _write_fake_prediction_files(Path(tmp_dir))
        blended = soft_vote_predictions(prediction_files)
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
        submission = build_submission_frame(blended, threshold)

        print("Blended probabilities")
        print(blended.write_csv())
        print()
        print(f"Best threshold: {threshold}")
        print(f"Validation Macro-F1: {macro_f1:.4f}")
        print()
        print("Submission preview")
        print(submission.write_csv())


if __name__ == "__main__":
    main()

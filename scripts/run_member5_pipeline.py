import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.post_processing import run_member5_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Member 5 soft voting, threshold optimization, and submission export."
    )
    parser.add_argument("--oof-predictions", nargs="+", required=True)
    parser.add_argument("--oof-labels", required=True)
    parser.add_argument("--test-predictions", nargs="*", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--probability-column", default=None)
    parser.add_argument("--min-threshold", type=float, default=0.10)
    parser.add_argument("--max-threshold", type=float, default=0.90)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--submission-filename", default="submission.csv")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="teknofest-eticaret-2026")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_member5_pipeline(
        oof_prediction_files=args.oof_predictions,
        oof_labels_file=args.oof_labels,
        output_dir=args.output_dir,
        test_prediction_files=args.test_predictions,
        id_column=args.id_column,
        label_column=args.label_column,
        probability_column=args.probability_column,
        min_threshold=args.min_threshold,
        max_threshold=args.max_threshold,
        step=args.step,
        submission_filename=args.submission_filename,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
    )

    print(f"Best threshold: {result.threshold}")
    print(f"Validation Macro-F1: {result.validation_macro_f1:.4f}")
    print(f"OOF blended: {result.oof_blended_path}")
    print(f"OOF scored: {result.oof_scored_path}")
    print(f"Metrics: {result.metrics_path}")
    if result.submission_path:
        print(f"Submission: {result.submission_path}")


if __name__ == "__main__":
    main()

import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl

from src.post_processing import (
    build_submission_frame,
    find_optimal_threshold,
    macro_f1_for_threshold,
    probabilities_to_labels,
    run_member5_pipeline,
    soft_vote_predictions,
)
from src.mlops import (
    build_default_run_config,
    finish_wandb_run,
    init_wandb_run,
    log_wandb_metrics,
)


class TestPostProcessing(unittest.TestCase):
    def test_find_optimal_threshold_prefers_best_macro_f1(self) -> None:
        probabilities = np.array([0.05, 0.20, 0.35, 0.45, 0.55, 0.62, 0.80, 0.95])
        labels = np.array([0, 0, 0, 1, 1, 1, 1, 1])

        threshold = find_optimal_threshold(probabilities, labels)
        score = macro_f1_for_threshold(probabilities, labels, threshold)

        self.assertEqual(threshold, 0.45)
        self.assertEqual(score, 1.0)

    def test_probabilities_to_labels_applies_threshold(self) -> None:
        probabilities = np.array([0.10, 0.49, 0.50, 0.91])

        labels = probabilities_to_labels(probabilities, threshold=0.50)

        np.testing.assert_array_equal(labels, np.array([0, 0, 1, 1]))

    def test_soft_vote_predictions_aligns_by_id_not_row_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            model_1 = pl.DataFrame(
                {"id": ["A", "B", "C"], "probability": [0.90, 0.10, 0.40]}
            )
            model_2 = pl.DataFrame(
                {"id": ["C", "B", "A"], "probability": [0.50, 0.30, 0.70]}
            )

            model_1_path = tmp_path / "model_1.csv"
            model_2_path = tmp_path / "model_2.csv"
            model_1.write_csv(model_1_path)
            model_2.write_csv(model_2_path)

            blended = soft_vote_predictions([str(model_1_path), str(model_2_path)])

            expected = {"A": 0.80, "B": 0.20, "C": 0.45}
            actual = dict(
                zip(
                    blended["id"].to_list(),
                    blended["blended_probability"].to_list(),
                    strict=True,
                )
            )

            self.assertEqual(set(actual), set(expected))
            for item_id, expected_probability in expected.items():
                self.assertAlmostEqual(actual[item_id], expected_probability)

    def test_build_submission_frame_outputs_binary_labels(self) -> None:
        blended = pl.DataFrame(
            {
                "id": ["A", "B", "C"],
                "blended_probability": [0.80, 0.20, 0.45],
            }
        )

        submission = build_submission_frame(blended, threshold=0.45)

        self.assertEqual(submission.columns, ["id", "label"])
        self.assertEqual(submission["label"].to_list(), [1, 0, 1])

    def test_member5_pipeline_writes_metrics_and_submission(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            oof_1 = tmp_path / "oof_model_1.csv"
            oof_2 = tmp_path / "oof_model_2.csv"
            labels_path = tmp_path / "labels.csv"
            test_1 = tmp_path / "test_model_1.csv"
            test_2 = tmp_path / "test_model_2.csv"
            output_dir = tmp_path / "outputs"

            pl.DataFrame(
                {"id": ["A", "B", "C", "D"], "probability": [0.10, 0.20, 0.70, 0.80]}
            ).write_csv(oof_1)
            pl.DataFrame(
                {"id": ["C", "A", "D", "B"], "probability": [0.60, 0.00, 0.90, 0.30]}
            ).write_csv(oof_2)
            pl.DataFrame(
                {"id": ["A", "B", "C", "D"], "label": [0, 0, 1, 1]}
            ).write_csv(labels_path)
            pl.DataFrame(
                {"id": ["T1", "T2"], "probability": [0.80, 0.20]}
            ).write_csv(test_1)
            pl.DataFrame(
                {"id": ["T2", "T1"], "probability": [0.30, 0.70]}
            ).write_csv(test_2)

            result = run_member5_pipeline(
                oof_prediction_files=[oof_1, oof_2],
                oof_labels_file=labels_path,
                test_prediction_files=[test_1, test_2],
                output_dir=output_dir,
                use_wandb=False,
            )

            self.assertEqual(result.threshold, 0.5)
            self.assertEqual(result.validation_macro_f1, 1.0)
            self.assertTrue(Path(result.metrics_path).exists())
            self.assertTrue(Path(result.oof_blended_path).exists())
            self.assertTrue(Path(result.oof_scored_path).exists())
            self.assertIsNotNone(result.submission_path)

            submission = pl.read_csv(result.submission_path)
            self.assertEqual(submission.columns, ["id", "label"])
            self.assertEqual(
                dict(zip(submission["id"], submission["label"], strict=True)),
                {"T1": 1, "T2": 0},
            )


class TestWandbTracking(unittest.TestCase):
    def test_wandb_helpers_can_be_disabled_for_local_smoke_tests(self) -> None:
        run = init_wandb_run(enabled=False)

        log_wandb_metrics(run, {"macro_f1": 0.75}, step=1)
        finish_wandb_run(run)

        self.assertIsNone(run)

    def test_default_run_config_accepts_extra_values(self) -> None:
        run_config = build_default_run_config({"fold": 2, "threshold": 0.45})

        self.assertEqual(run_config["fold"], 2)
        self.assertEqual(run_config["threshold"], 0.45)
        self.assertIn("MODEL_NAME", run_config)


if __name__ == "__main__":
    unittest.main()

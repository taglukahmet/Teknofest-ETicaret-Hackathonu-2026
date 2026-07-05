import numpy as np
from sklearn.metrics import f1_score

from src.config import config


def probabilities_to_labels(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    """
    Converts class-1 probabilities into binary labels with a fixed threshold.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    if probabilities.size == 0:
        raise ValueError("probabilities cannot be empty.")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities must contain only finite numeric values.")

    return (probabilities >= threshold).astype(int)


def macro_f1_for_threshold(
    probabilities: np.ndarray,
    true_labels: np.ndarray,
    threshold: float,
) -> float:
    """
    Scores one threshold with the competition metric: Macro-F1.
    """
    labels = np.asarray(true_labels, dtype=int).reshape(-1)
    predictions = probabilities_to_labels(probabilities, threshold)

    if predictions.shape[0] != labels.shape[0]:
        raise ValueError("probabilities and true_labels must have the same length.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("true_labels must contain only 0 and 1 values.")

    return float(f1_score(labels, predictions, average="macro", zero_division=0))


def find_optimal_threshold(
    oof_probabilities: np.ndarray,
    true_labels: np.ndarray,
    min_threshold: float = 0.10,
    max_threshold: float = 0.90,
    step: float = 0.01,
) -> float:
    """
    (Task: Member 5)
    Searches a threshold range to find the value that maximizes Macro-F1.

    Args:
        oof_probabilities: Model probability scores for class 1 (VAR).
        true_labels: Ground-truth labels, encoded as 0 (YOK) or 1 (VAR).
        min_threshold: Lowest threshold to test.
        max_threshold: Highest threshold to test.
        step: Distance between tested thresholds.

    Returns:
        The threshold that gives the best Macro-F1 score.
    """
    probabilities = np.asarray(oof_probabilities, dtype=float).reshape(-1)
    labels = np.asarray(true_labels, dtype=int).reshape(-1)

    if probabilities.shape[0] != labels.shape[0]:
        raise ValueError("oof_probabilities and true_labels must have the same length.")
    if probabilities.size == 0:
        raise ValueError("oof_probabilities and true_labels cannot be empty.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("true_labels must contain only 0 and 1 values.")
    if not 0 <= min_threshold <= max_threshold <= 1:
        raise ValueError("threshold bounds must satisfy 0 <= min <= max <= 1.")
    if step <= 0:
        raise ValueError("step must be greater than 0.")

    thresholds = np.arange(min_threshold, max_threshold + step / 2, step)

    best_threshold = config.DEFAULT_THRESHOLD
    best_score = -1.0
    best_distance_from_default = float("inf")

    for threshold in thresholds:
        score = macro_f1_for_threshold(probabilities, labels, float(threshold))

        distance_from_default = abs(float(threshold) - config.DEFAULT_THRESHOLD)
        is_better = score > best_score
        is_stable_tie = np.isclose(score, best_score) and (
            distance_from_default < best_distance_from_default
        )

        if is_better or is_stable_tie:
            best_score = score
            best_threshold = float(threshold)
            best_distance_from_default = distance_from_default

    return round(best_threshold, 4)

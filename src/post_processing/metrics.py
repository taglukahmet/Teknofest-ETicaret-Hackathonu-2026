import numpy as np

def find_optimal_threshold(oof_probabilities: np.ndarray, true_labels: np.ndarray) -> float:
    """
    (Task: Member 5)
    Searches from 0.10 to 0.90 to find the decimal that maximizes the Macro-F1 score.
    Returns the optimal float.
    """
    return 0.50
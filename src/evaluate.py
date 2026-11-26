"""Evaluation helpers for the previously selected educational baseline."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor
from src.train import SEED

THRESHOLD = 0.5


def build_selected_model() -> Pipeline:
    """Recreate the best F1 candidate from the training-only search."""
    return Pipeline(
        [
            ("preprocess", build_preprocessor()),
            (
                "model",
                LogisticRegression(
                    C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED
                ),
            ),
        ]
    )


def evaluate_predictions(y_true, scores, threshold=THRESHOLD):
    """Calculate binary metrics from fixed model scores and a fixed threshold."""
    labels = np.asarray(y_true)
    probabilities = np.asarray(scores, dtype=float)
    if labels.ndim != 1 or probabilities.ndim != 1 or len(labels) != len(probabilities):
        raise ValueError("Мітки і scores повинні бути одновимірними та однакової довжини.")
    if set(labels) != {0, 1}:
        raise ValueError("Для оцінювання потрібні обидва класи: 0 і 1.")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Поріг повинен бути скінченним числом від 0 до 1.")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Scores повинні бути скінченними числами від 0 до 1.")

    predicted = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    metrics = {
        "precision": precision_score(labels, predicted, zero_division=0),
        "recall": recall_score(labels, predicted, zero_division=0),
        "f1": f1_score(labels, predicted, zero_division=0),
        "roc_auc": roc_auc_score(labels, probabilities),
        "average_precision": average_precision_score(labels, probabilities),
    }
    return metrics, matrix

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)
from loguru import logger


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                        y_proba: np.ndarray, model_name: str) -> dict:
    """
    Computes a comprehensive set of metrics for binary classification.
    Returns a flat dict suitable for CSV/JSON export.
    """
    metrics = {
        "model": model_name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_score": round(f1_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
        "pr_auc": round(average_precision_score(y_true, y_proba), 4),
    }

    # Business metric: revenue lift
    # Top 20% by predicted probability — what % of actual buyers captured?
    top_20_threshold = np.percentile(y_proba, 80)
    top_20_mask = y_proba >= top_20_threshold
    lift = y_true[top_20_mask].sum() / max(y_true.sum(), 1)
    metrics["lift_top20pct"] = round(float(lift), 4)

    logger.info(
        f"[{model_name}] AUC: {metrics['roc_auc']} | "
        f"F1: {metrics['f1_score']} | "
        f"Lift@20%: {metrics['lift_top20pct']}"
    )

    return metrics


def compare_models(results: list[dict]) -> pd.DataFrame:
    """Converts list of metric dicts into a ranked comparison DataFrame."""
    df = pd.DataFrame(results)
    df = df.sort_values("roc_auc", ascending=False).reset_index(drop=True)
    df.index += 1
    df.index.name = "rank"
    return df
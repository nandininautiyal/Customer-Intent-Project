import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
)
from loguru import logger


def compute_all_metrics(
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    y_proba:    np.ndarray,
    model_name: str
) -> dict:
    """
    Computes a comprehensive set of metrics for binary classification.
    Returns a flat dict suitable for CSV / JSON export.

    Metrics:
        accuracy   — overall correct predictions
        f1_score   — harmonic mean of precision and recall
        precision  — of predicted buyers, how many actually bought
        recall     — of actual buyers, how many did we catch
        roc_auc    — ranking quality (threshold-independent)
        pr_auc     — precision-recall AUC (better for imbalanced data)
        mcc        — Matthews Correlation Coefficient
                     ranges -1 to 1, best single metric for imbalanced data
        lift_top20 — % of actual buyers captured in top 20% by probability
    """
    # --- Standard classification metrics ---
    accuracy  = round(accuracy_score(y_true, y_pred), 4)
    f1        = round(f1_score(y_true, y_pred, zero_division=0), 4)
    precision = round(precision_score(y_true, y_pred, zero_division=0), 4)
    recall    = round(recall_score(y_true, y_pred, zero_division=0), 4)
    roc_auc   = round(roc_auc_score(y_true, y_proba), 4)
    pr_auc    = round(average_precision_score(y_true, y_proba), 4)
    mcc       = round(matthews_corrcoef(y_true, y_pred), 4)

    # --- Business metric: Lift @ Top 20% ---
    # How many actual buyers are captured if we target
    # the top 20% of sessions ranked by predicted probability?
    top20_threshold = np.percentile(y_proba, 80)
    top20_mask      = y_proba >= top20_threshold
    lift_top20      = round(
        float(y_true[top20_mask].sum() / max(y_true.sum(), 1)), 4
    )

    metrics = {
        "model":         model_name,
        "accuracy":      accuracy,
        "f1_score":      f1,
        "precision":     precision,
        "recall":        recall,
        "roc_auc":       roc_auc,
        "pr_auc":        pr_auc,
        "mcc":           mcc,
        "lift_top20pct": lift_top20,
    }

    logger.info(
        f"[{model_name}] "
        f"Acc: {accuracy} | "
        f"F1: {f1} | "
        f"AUC: {roc_auc} | "
        f"MCC: {mcc} | "
        f"Lift@20%: {lift_top20}"
    )

    return metrics


def compare_models(results: list) -> pd.DataFrame:
    """
    Converts list of metric dicts into a ranked comparison DataFrame.
    Ranked by ROC-AUC descending.
    """
    df = pd.DataFrame(results)
    df = df.sort_values("roc_auc", ascending=False).reset_index(drop=True)
    df.index += 1
    df.index.name = "rank"
    return df
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from pathlib import Path
from loguru import logger

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "Logistic": "#4C72B0",
    "XGBoost": "#DD8452",
    "NeuralNet": "#55A868",
    "Ensemble": "#C44E52",
}


def save_metrics_csv(metrics_df: pd.DataFrame):
    path = REPORTS_DIR / "model_comparison_metrics.csv"
    metrics_df.to_csv(path)
    logger.info(f"Metrics saved: {path}")


def save_metrics_json(metrics_df: pd.DataFrame):
    path = REPORTS_DIR / "model_comparison_metrics.json"
    metrics_df.to_json(path, orient="records", indent=2)
    logger.info(f"Metrics saved: {path}")


def plot_roc_curves(results: list, y_true: np.ndarray):
    """Plots ROC curves for all models on one chart."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for r in results:
        fpr, tpr, _ = roc_curve(y_true, r["proba"])
        ax.plot(
            fpr, tpr,
            label=f"{r['model']} (AUC={r['roc_auc']:.3f})",
            color=PALETTE.get(r["model"], "gray"),
            linewidth=2
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "roc_curves.png", dpi=150)
    plt.close()
    logger.info("ROC curves saved.")


def plot_pr_curves(results: list, y_true: np.ndarray):
    """Plots Precision-Recall curves (better for imbalanced datasets)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for r in results:
        prec, rec, _ = precision_recall_curve(y_true, r["proba"])
        ax.plot(
            rec, prec,
            label=f"{r['model']} (PR-AUC={r['pr_auc']:.3f})",
            color=PALETTE.get(r["model"], "gray"),
            linewidth=2
        )

    baseline = y_true.mean()
    ax.axhline(baseline, color="k", linestyle="--", label=f"Baseline ({baseline:.2f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "pr_curves.png", dpi=150)
    plt.close()
    logger.info("PR curves saved.")


def plot_confusion_matrices(results: list, y_true: np.ndarray):
    """Plots confusion matrices for all models in a grid."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))

    for ax, r in zip(axes, results):
        cm = confusion_matrix(y_true, r["pred"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Buy", "Buy"],
            yticklabels=["No Buy", "Buy"],
            ax=ax
        )
        ax.set_title(r["model"], fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Confusion matrices saved.")


def plot_metrics_bar(metrics_df: pd.DataFrame):
    """Bar chart comparing key metrics across models."""
    plot_cols = ["accuracy", "f1_score", "roc_auc", "pr_auc", "lift_top20pct"]
    df_plot = metrics_df.set_index("model")[plot_cols]

    ax = df_plot.plot(
        kind="bar", figsize=(12, 6),
        color=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"],
        edgecolor="white", linewidth=0.5
    )
    ax.set_title("Model Comparison — Key Metrics", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "metrics_comparison.png", dpi=150)
    plt.close()
    logger.info("Metrics bar chart saved.")


def generate_all_reports(results: list, metrics_df: pd.DataFrame, y_true: np.ndarray):
    """Master function to generate all visual and data reports."""
    save_metrics_csv(metrics_df)
    save_metrics_json(metrics_df)
    plot_roc_curves(results, y_true)
    plot_pr_curves(results, y_true)
    plot_confusion_matrices(results, y_true)
    plot_metrics_bar(metrics_df)
    logger.info("All reports generated in reports/")
import shap
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger


REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def explain_xgboost(model, X_test: np.ndarray, feature_names: list):
    """
    Generates SHAP summary and waterfall plots for XGBoost model.
    Saves to reports/shap_summary.png
    """
    logger.info("[SHAP] Computing SHAP values for XGBoost...")

    explainer = shap.TreeExplainer(model.model)
    shap_values = explainer.shap_values(X_test)

    # Summary plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_test,
        feature_names=feature_names,
        show=False,
        plot_type="bar"
    )
    plt.title("SHAP Feature Importance — XGBoost", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_summary_bar.png", dpi=150)
    plt.close()

    # Beeswarm plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_test,
        feature_names=feature_names,
        show=False
    )
    plt.title("SHAP Value Distribution — XGBoost", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_summary_beeswarm.png", dpi=150)
    plt.close()

    logger.info("[SHAP] Plots saved to reports/")
    return shap_values
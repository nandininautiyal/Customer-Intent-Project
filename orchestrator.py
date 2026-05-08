"""
Master orchestrator — runs the full pipeline end to end:
    1. Ingest raw data
    2. Engineer features
    3. Preprocess & split (with SMOTE)
    4. Train all base models (with tuning + threshold optimization)
    5. Train stacking ensemble (OOF meta-features)
    6. Evaluate on test set
    7. Generate all reports + SHAP plots
    8. Save all model artifacts
"""

from pathlib import Path
import numpy as np
from loguru import logger

from pipeline.ingest import load_raw_data
from pipeline.features import engineer_features
from pipeline.preprocess import preprocess

from models.logistic_model import LogisticIntentModel
from models.xgboost_model import XGBoostIntentModel
from models.neural_model import NeuralIntentModel
from models.ensemble_model import EnsembleIntentModel

from evaluation.metrics import compute_all_metrics, compare_models
from evaluation.shap_explainer import explain_xgboost
from evaluation.reporter import generate_all_reports

ARTIFACTS_DIR = Path("reports/artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def run():
    logger.info("=" * 60)
    logger.info("Customer Intent Engine — Full Pipeline Start")
    logger.info("=" * 60)

    # --- 1. Ingest ---
    df = load_raw_data()

    # --- 2. Feature Engineering ---
    df = engineer_features(df)

    # --- 3. Preprocess (returns numpy arrays, SMOTE applied on train only) ---
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     feature_names) = preprocess(df)

    # Guarantee numpy arrays throughout (defensive cast)
    X_train = np.array(X_train)
    X_val   = np.array(X_val)
    X_test  = np.array(X_test)
    y_train = np.array(y_train)
    y_val   = np.array(y_val)
    y_test  = np.array(y_test)

    input_dim = X_train.shape[1]
    logger.info(f"Input dimension: {input_dim}")
    logger.info(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    # --- 4. Train base models ---
    logger.info("-" * 40)
    logger.info("Training Logistic Regression...")
    logistic = LogisticIntentModel()
    logistic.train(X_train, y_train, X_val, y_val)
    logistic.save(str(ARTIFACTS_DIR / "logistic.pkl"))

    logger.info("-" * 40)
    logger.info("Training XGBoost (Optuna tuning)...")
    xgb = XGBoostIntentModel(tune=True, n_trials=40)
    xgb.train(X_train, y_train, X_val, y_val)
    xgb.save(str(ARTIFACTS_DIR / "xgboost.pkl"))

    logger.info("-" * 40)
    logger.info("Training Neural Network...")
    neural = NeuralIntentModel(input_dim=input_dim, epochs=50)
    neural.train(X_train, y_train, X_val, y_val)
    neural.save(str(ARTIFACTS_DIR / "neural.pth"))

    # --- 5. Train stacking ensemble (OOF) ---
    logger.info("-" * 40)
    logger.info("Training Stacking Ensemble (OOF)...")
    ensemble = EnsembleIntentModel([logistic, xgb, neural])
    ensemble.train(X_train, y_train, X_val, y_val)
    ensemble.save(str(ARTIFACTS_DIR / "ensemble_meta.pkl"))

    # --- 6. Evaluate all models on held-out test set ---
    logger.info("-" * 40)
    logger.info("Evaluating on test set...")

    model_map = {
        "Logistic":  logistic,
        "XGBoost":   xgb,
        "NeuralNet": neural,
        "Ensemble":  ensemble,
    }

    results      = []
    metrics_list = []

    for name, model in model_map.items():
        pred  = model.predict(X_test)
        proba = model.predict_proba(X_test)
        m = compute_all_metrics(y_test, pred, proba, name)
        # Store pred/proba for plotting but not in CSV
        results.append({**m, "pred": pred, "proba": proba})
        metrics_list.append({k: v for k, v in m.items()})

    # --- 7. Reports ---
    logger.info("-" * 40)
    metrics_df = compare_models(metrics_list)
    logger.info("\n" + metrics_df.to_string())

    generate_all_reports(results, metrics_df, y_test)

    # --- 8. SHAP ---
    logger.info("Generating SHAP explanations...")
    explain_xgboost(xgb, X_test, feature_names)

    logger.info("=" * 60)
    logger.info("Pipeline complete. Check the reports/ folder.")
    logger.info("=" * 60)


if __name__ == "__main__":
    run()
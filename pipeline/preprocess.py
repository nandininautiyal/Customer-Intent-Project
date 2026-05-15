import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.combine import SMOTETomek
from loguru import logger
import joblib
from pathlib import Path


ARTIFACTS_DIR = Path("reports/artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def preprocess(df: pd.DataFrame):
    """
    Full preprocessing pipeline:
    - Encodes categoricals
    - Scales numerics
    - Applies SMOTETomek on training set only:
        SMOTE oversamples minority class synthetically
        Tomek Links removes borderline noisy samples
        Together they oversample AND clean the decision boundary
    - Splits into train/val/test (60/20/20)

    Returns:
        X_train_bal, X_val, X_test,
        y_train_bal, y_val, y_test,
        feature_names
    """
    df = df.copy()

    # --- Encode binary ---
    df["Weekend"] = df["Weekend"].astype(int)
    df["Revenue"] = df["Revenue"].astype(int)

    # --- Encode Month ---
    month_map = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
        "May": 5, "June": 6, "Jul": 7, "Aug": 8,
        "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
    }
    df["Month"] = df["Month"].map(month_map).fillna(0).astype(int)

    # --- Encode VisitorType ---
    le = LabelEncoder()
    df["VisitorType"] = le.fit_transform(df["VisitorType"])
    joblib.dump(le, ARTIFACTS_DIR / "visitor_type_encoder.pkl")

    # --- Define features and target ---
    target = "Revenue"
    features = [col for col in df.columns if col != target]
    X = df[features]
    y = df[target]
    feature_names = features

    logger.info(f"Features: {len(feature_names)} | Target: '{target}'")
    logger.info(
        f"Class balance before resampling — "
        f"Buy: {y.sum()} | No Buy: {(y == 0).sum()}"
    )

    # --- Train / Val / Test split (60/20/20) ---
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    logger.info(
        f"Split — Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}"
    )

    # --- Scale BEFORE resampling (fit only on train) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.pkl")
    logger.info("Scaler saved.")

    # --- SMOTETomek on training set only ---
    # CRITICAL: never apply to val or test — that is data leakage
    smt = SMOTETomek(random_state=42)
    X_train_bal, y_train_bal = smt.fit_resample(X_train_scaled, y_train)

    logger.info(
        f"After SMOTETomek — "
        f"Buy: {y_train_bal.sum()} | "
        f"No Buy: {(y_train_bal == 0).sum()} | "
        f"Total: {len(y_train_bal)}"
    )

    return (
        X_train_bal, X_val_scaled, X_test_scaled,
        y_train_bal, y_val.values, y_test.values,
        feature_names
    )
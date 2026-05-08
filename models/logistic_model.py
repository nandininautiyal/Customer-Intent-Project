import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from loguru import logger
from models.base_model import BaseIntentModel


class LogisticIntentModel(BaseIntentModel):
    """
    Logistic Regression baseline with threshold tuning.
    class_weight=balanced handles any residual imbalance after SMOTE.
    """

    def __init__(self):
        self.model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=42
        )
        self.optimal_threshold = 0.5

    def _tune_threshold(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        probas = self.model.predict_proba(X_val)[:, 1]
        best_thresh, best_f1 = 0.5, 0.0
        for thresh in np.arange(0.20, 0.81, 0.01):
            preds = (probas >= thresh).astype(int)
            score = f1_score(y_val, preds, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_thresh = thresh
        logger.info(f"[Logistic] Optimal threshold: {best_thresh:.2f} → F1: {best_f1:.4f}")
        return float(best_thresh)

    def train(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, y_train)
        self.optimal_threshold = self._tune_threshold(X_val, y_val)
        val_acc = self.model.score(X_val, y_val)
        logger.info(f"[Logistic] Val Accuracy: {val_acc:.4f}")

    def predict(self, X):
        return (self.predict_proba(X) >= self.optimal_threshold).astype(int)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def save(self, path):
        joblib.dump({"model": self.model, "threshold": self.optimal_threshold}, path)
        logger.info(f"[Logistic] Saved to {path}")

    def load(self, path):
        data = joblib.load(path)
        self.model = data["model"]
        self.optimal_threshold = data.get("threshold", 0.5)
        logger.info(f"[Logistic] Loaded. Threshold: {self.optimal_threshold:.2f}")
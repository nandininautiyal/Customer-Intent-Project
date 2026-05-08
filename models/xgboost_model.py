import numpy as np
import joblib
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from loguru import logger
from models.base_model import BaseIntentModel

optuna.logging.set_verbosity(optuna.logging.WARNING)


class XGBoostIntentModel(BaseIntentModel):
    """
    XGBoost with Optuna tuning + automatic F1-optimal threshold tuning.
    scale_pos_weight removed — SMOTE handles imbalance at data level now.
    """

    def __init__(self, tune: bool = True, n_trials: int = 50):
        self.tune = tune
        self.n_trials = n_trials
        self.model = None
        self.best_params = {}
        self.optimal_threshold = 0.5

    def _objective(self, trial, X_train, y_train, X_val, y_val):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 700),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 3),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 2),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 3),
            "eval_metric": "logloss",
            "random_state": 42,
            "tree_method": "hist",
        }
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # Tune threshold per trial too
        probas = model.predict_proba(X_val)[:, 1]
        best_f1 = 0.0
        for thresh in np.arange(0.25, 0.75, 0.01):
            preds = (probas >= thresh).astype(int)
            score = f1_score(y_val, preds, zero_division=0)
            best_f1 = max(best_f1, score)

        return best_f1

    def _tune_threshold(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        probas = self.model.predict_proba(X_val)[:, 1]
        best_thresh, best_f1 = 0.5, 0.0
        for thresh in np.arange(0.20, 0.81, 0.01):
            preds = (probas >= thresh).astype(int)
            score = f1_score(y_val, preds, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_thresh = thresh
        logger.info(f"[XGBoost] Optimal threshold: {best_thresh:.2f} → F1: {best_f1:.4f}")
        return float(best_thresh)

    def train(self, X_train, y_train, X_val, y_val):
        if self.tune:
            logger.info(f"[XGBoost] Optuna search — {self.n_trials} trials...")
            study = optuna.create_study(direction="maximize")
            study.optimize(
                lambda trial: self._objective(trial, X_train, y_train, X_val, y_val),
                n_trials=self.n_trials,
                show_progress_bar=False
            )
            self.best_params = study.best_params
            logger.info(f"[XGBoost] Best params: {self.best_params}")
        else:
            self.best_params = {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
            }

        self.model = XGBClassifier(
            **self.best_params,
            eval_metric="logloss",
            random_state=42,
            tree_method="hist"
        )
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        self.optimal_threshold = self._tune_threshold(X_val, y_val)

    def predict(self, X):
        return (self.predict_proba(X) >= self.optimal_threshold).astype(int)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def save(self, path):
        joblib.dump({"model": self.model, "threshold": self.optimal_threshold}, path)
        logger.info(f"[XGBoost] Saved to {path}")

    def load(self, path):
        data = joblib.load(path)
        self.model = data["model"]
        self.optimal_threshold = data.get("threshold", 0.5)
        logger.info(f"[XGBoost] Loaded. Threshold: {self.optimal_threshold:.2f}")
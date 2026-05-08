import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from loguru import logger
from models.base_model import BaseIntentModel


class EnsembleIntentModel(BaseIntentModel):
    """
    Stacking Ensemble with out-of-fold (OOF) meta-feature generation.

    Why OOF instead of val-set-only:
    - Training meta-learner on val set = ~2400 samples
    - OOF gives meta-learner ALL training data = ~7200 samples
    - More data for meta-learner = better calibration = better F1

    Architecture:
        Layer 1 (5-fold OOF):  [Logistic, XGBoost, NeuralNet]
                                → generates meta-features for every train sample
        Layer 2 (meta-learner): Logistic Regression trained on OOF meta-features
        Inference:              Base models predict on new data → meta-learner combines
    """

    def __init__(self, base_models: list, n_folds: int = 5):
        self.base_models = base_models
        self.n_folds = n_folds
        self.meta_learner = LogisticRegression(
            C=0.5,
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
        self.optimal_threshold = 0.5

    def _get_init_params(self, model) -> dict:
        """Returns safe init params for each model type for OOF copying."""
        from models.logistic_model import LogisticIntentModel
        from models.xgboost_model import XGBoostIntentModel
        from models.neural_model import NeuralIntentModel

        if isinstance(model, LogisticIntentModel):
            return {}
        elif isinstance(model, XGBoostIntentModel):
            # Reuse already-tuned params, skip re-tuning in OOF folds
            return {"tune": False}
        elif isinstance(model, NeuralIntentModel):
            return {
                "input_dim": model.input_dim,
                "epochs": 30,
                "lr": model.lr,
                "batch_size": model.batch_size
            }
        return {}

    def _generate_oof_features(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> np.ndarray:
        """
        Generates out-of-fold predictions from each base model.
        Each base model predicts on held-out folds it never trained on.
        Result is a (n_samples, n_models) matrix of probabilities.
        """
        n_samples = X_train.shape[0]
        n_models = len(self.base_models)
        oof_preds = np.zeros((n_samples, n_models))

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        # Guarantee numpy — prevents pandas Series from reaching PyTorch
        y_train = np.array(y_train)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            logger.info(f"[Ensemble] OOF Fold {fold + 1}/{self.n_folds}")

            X_fold_train = X_train[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val   = X_train[val_idx]
            y_fold_val   = y_train[val_idx]

            for i, model in enumerate(self.base_models):
                model_copy = model.__class__(**self._get_init_params(model))
                model_copy.train(X_fold_train, y_fold_train, X_fold_val, y_fold_val)
                oof_preds[val_idx, i] = model_copy.predict_proba(X_fold_val)

        return oof_preds

    def _stack(self, X: np.ndarray) -> np.ndarray:
        """Inference-time stacking: collect probas from all base models."""
        return np.column_stack([
            model.predict_proba(X) for model in self.base_models
        ])

    def _tune_threshold(self, meta_X: np.ndarray, y_true: np.ndarray) -> float:
        """
        Finds the probability threshold that maximizes F1 score.
        Tests thresholds from 0.20 to 0.80 in steps of 0.01.
        """
        probas = self.meta_learner.predict_proba(meta_X)[:, 1]
        best_thresh = 0.5
        best_f1 = 0.0

        for thresh in np.arange(0.20, 0.81, 0.01):
            preds = (probas >= thresh).astype(int)
            score = f1_score(y_true, preds, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_thresh = thresh

        logger.info(
            f"[Ensemble] Optimal threshold: {best_thresh:.2f} → F1: {best_f1:.4f}"
        )
        return float(best_thresh)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> None:
        """
        Full stacking train:
        1. Generate OOF meta-features from base models across 5 folds
        2. Train meta-learner on OOF features (no leakage)
        3. Tune decision threshold on clean validation set
        """
        # Guarantee numpy throughout
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_val   = np.array(X_val)
        y_val   = np.array(y_val)

        logger.info("[Ensemble] Generating OOF meta-features...")
        oof_meta_X = self._generate_oof_features(X_train, y_train)

        logger.info("[Ensemble] Training meta-learner on OOF features...")
        self.meta_learner.fit(oof_meta_X, y_train)

        # Tune threshold on clean val set (no leakage)
        val_meta_X = self._stack(X_val)
        self.optimal_threshold = self._tune_threshold(val_meta_X, y_val)

        val_acc = self.meta_learner.score(val_meta_X, y_val)
        logger.info(f"[Ensemble] Meta-learner Val Accuracy: {val_acc:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Uses tuned threshold instead of fixed 0.5."""
        proba = self.predict_proba(X)
        return (proba >= self.optimal_threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        meta_X = self._stack(X)
        return self.meta_learner.predict_proba(meta_X)[:, 1]

    def save(self, path: str) -> None:
        joblib.dump(
            {
                "meta_learner": self.meta_learner,
                "threshold": self.optimal_threshold
            },
            path
        )
        logger.info(f"[Ensemble] Saved to {path}")

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.meta_learner = data["meta_learner"]
        self.optimal_threshold = data.get("threshold", 0.5)
        logger.info(
            f"[Ensemble] Loaded. Threshold: {self.optimal_threshold:.2f}"
        )
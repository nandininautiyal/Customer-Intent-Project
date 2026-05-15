import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, fbeta_score
from loguru import logger
from models.base_model import BaseIntentModel


class EnsembleIntentModel(BaseIntentModel):
    """
    Stacking Ensemble with out-of-fold (OOF) meta-feature generation
    and an XGBoost meta-learner.

    Improvements over basic stacking:
    1. XGBoost meta-learner — learns non-linear combinations of base models
    2. Extended meta-features — includes original features alongside base
       model probabilities so the meta-learner has richer context
    3. Probability calibration — adds difference and product of base model
       probabilities as extra meta-features
    4. F-beta threshold tuning — beta=0.8 balances precision and recall
       without over-optimising recall at the cost of precision
    5. OOF guarantee — zero data leakage across all 5 folds
    """

    def __init__(self, base_models: list, n_folds: int = 5):
        self.base_models = base_models
        self.n_folds     = n_folds

        self.meta_learner = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.5,
            reg_lambda=1.5,
            scale_pos_weight=2,
            eval_metric="logloss",
            random_state=42,
            tree_method="hist",
        )
        self.optimal_threshold = 0.5

    # ------------------------------------------------------------------
    # Init param helpers for OOF fold copies
    # ------------------------------------------------------------------

    def _get_init_params(self, model) -> dict:
        from models.logistic_model import LogisticIntentModel
        from models.xgboost_model import XGBoostIntentModel
        from models.neural_model import NeuralIntentModel

        if isinstance(model, LogisticIntentModel):
            return {}
        elif isinstance(model, XGBoostIntentModel):
            return {"tune": False}
        elif isinstance(model, NeuralIntentModel):
            return {
                "input_dim":  model.input_dim,
                "epochs":     30,
                "lr":         model.lr,
                "batch_size": model.batch_size,
            }
        return {}

    # ------------------------------------------------------------------
    # OOF meta-feature generation
    # ------------------------------------------------------------------

    def _generate_oof_features(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> np.ndarray:
        """
        Generates out-of-fold probability predictions from each base model.
        No base model ever predicts on data it was trained on.

        Returns: (n_samples, n_models) probability matrix
        """
        n_samples = X_train.shape[0]
        n_models  = len(self.base_models)
        oof_preds = np.zeros((n_samples, n_models))

        skf     = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=42
        )
        y_train = np.array(y_train)

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(X_train, y_train)
        ):
            logger.info(f"[Ensemble] OOF Fold {fold + 1}/{self.n_folds}")

            X_fold_train = X_train[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val   = X_train[val_idx]
            y_fold_val   = y_train[val_idx]

            for i, model in enumerate(self.base_models):
                model_copy = model.__class__(
                    **self._get_init_params(model)
                )
                model_copy.train(
                    X_fold_train, y_fold_train,
                    X_fold_val,   y_fold_val,
                )
                oof_preds[val_idx, i] = model_copy.predict_proba(
                    X_fold_val
                )

        return oof_preds

    # ------------------------------------------------------------------
    # Meta-feature construction
    # ------------------------------------------------------------------

    def _build_meta_features(
        self,
        base_probas: np.ndarray,
        X_original:  np.ndarray,
    ) -> np.ndarray:
        """
        Constructs an enriched meta-feature matrix by combining:

        1. Raw base model probabilities         [n_models columns]
           Each model's probability output

        2. Interaction features                 [3 columns]
           - Mean probability across all models
           - Std deviation (disagreement signal)
           - Max minus min (spread signal)

        3. Pairwise products                    [n_pairs columns]
           Product of each pair of model probabilities
           High product = all models agree on high intent

        4. Top-5 original features              [5 columns]
           PageValues, ExitBounceRisk, ProductPageRatio,
           AvgTimePerPage, VisitorType
           These are the SHAP-confirmed top drivers —
           giving the meta-learner direct access to raw
           signals prevents information loss from the
           base model bottleneck

        Total: n_models + 3 + n_pairs + 5 columns
        """
        n = base_probas.shape[0]

        # 1. Raw probabilities
        parts = [base_probas]

        # 2. Aggregate interaction features
        mean_p = base_probas.mean(axis=1, keepdims=True)
        std_p  = base_probas.std(axis=1,  keepdims=True)
        spread = (
            base_probas.max(axis=1, keepdims=True) -
            base_probas.min(axis=1, keepdims=True)
        )
        parts.append(mean_p)
        parts.append(std_p)
        parts.append(spread)

        # 3. Pairwise products
        n_models = base_probas.shape[1]
        for i in range(n_models):
            for j in range(i + 1, n_models):
                product = (
                    base_probas[:, i] * base_probas[:, j]
                ).reshape(-1, 1)
                parts.append(product)

        # 4. Top-5 SHAP-confirmed original features
        # Indices in the 24-feature vector (from pipeline/features.py):
        # 8  = PageValues
        # 23 = ExitBounceRisk
        # 19 = ProductPageRatio
        # 20 = AvgTimePerPage
        # 16 = VisitorType (encoded)
        top_feature_indices = [8, 23, 19, 20, 16]
        if X_original.shape[1] > max(top_feature_indices):
            parts.append(X_original[:, top_feature_indices])

        return np.hstack(parts)

    def _stack(self, X: np.ndarray) -> np.ndarray:
        """Inference-time: collect base probas and build meta-features."""
        base_probas = np.column_stack([
            model.predict_proba(X) for model in self.base_models
        ])
        return self._build_meta_features(base_probas, X)

    # ------------------------------------------------------------------
    # Threshold tuning
    # ------------------------------------------------------------------

    def _tune_threshold(
        self,
        meta_X: np.ndarray,
        y_true: np.ndarray,
    ) -> float:
        """
        Finds the threshold maximising F-beta (beta=0.8).
        beta < 1 weights precision slightly more than recall,
        preventing the ensemble from becoming recall-only.
        """
        probas      = self.meta_learner.predict_proba(meta_X)[:, 1]
        best_thresh = 0.5
        best_score  = 0.0

        for thresh in np.arange(0.20, 0.81, 0.01):
            preds = (probas >= thresh).astype(int)
            score = fbeta_score(
                y_true, preds, beta=0.8, zero_division=0
            )
            if score > best_score:
                best_score  = score
                best_thresh = thresh

        # Also log the F1 at this threshold for reference
        preds_at_best = (probas >= best_thresh).astype(int)
        f1_at_best    = f1_score(y_true, preds_at_best, zero_division=0)

        logger.info(
            f"[Ensemble] Optimal threshold: {best_thresh:.2f} | "
            f"F-beta(0.8): {best_score:.4f} | "
            f"F1: {f1_at_best:.4f}"
        )
        return float(best_thresh)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray,
        y_val:   np.ndarray,
    ) -> None:
        """
        Full stacking train:
        1. Generate OOF base model probabilities (5-fold, no leakage)
        2. Build enriched meta-features from OOF probas + original features
        3. Train XGBoost meta-learner
        4. Tune F-beta threshold on clean validation set
        """
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_val   = np.array(X_val)
        y_val   = np.array(y_val)

        # Step 1 — OOF base probabilities
        logger.info("[Ensemble] Generating OOF meta-features...")
        oof_base_probas = self._generate_oof_features(X_train, y_train)

        # Step 2 — Enrich with interaction features + top original features
        oof_meta_X = self._build_meta_features(oof_base_probas, X_train)
        val_meta_X = self._stack(X_val)

        logger.info(
            f"[Ensemble] Meta-feature matrix shape: {oof_meta_X.shape} "
            f"({oof_meta_X.shape[1]} features per sample)"
        )

        # Step 3 — Train meta-learner
        logger.info("[Ensemble] Training XGBoost meta-learner...")
        self.meta_learner.fit(
            oof_meta_X, y_train,
            eval_set=[(val_meta_X, y_val)],
            verbose=False,
        )

        # Step 4 — Tune threshold on clean val set
        self.optimal_threshold = self._tune_threshold(val_meta_X, y_val)

        # Final val F1 report
        probas    = self.meta_learner.predict_proba(val_meta_X)[:, 1]
        preds     = (probas >= self.optimal_threshold).astype(int)
        val_f1    = f1_score(y_val, preds, zero_division=0)
        val_acc   = (preds == y_val).mean()
        logger.info(
            f"[Ensemble] Val Accuracy: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (
            self.predict_proba(X) >= self.optimal_threshold
        ).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        meta_X = self._stack(X)
        return self.meta_learner.predict_proba(meta_X)[:, 1]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        joblib.dump(
            {
                "meta_learner": self.meta_learner,
                "threshold":    self.optimal_threshold,
            },
            path,
        )
        logger.info(f"[Ensemble] Saved to {path}")

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.meta_learner      = data["meta_learner"]
        self.optimal_threshold = data.get("threshold", 0.5)
        logger.info(
            f"[Ensemble] Loaded. Threshold: {self.optimal_threshold:.2f}"
        )
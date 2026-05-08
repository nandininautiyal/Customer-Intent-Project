from abc import ABC, abstractmethod
import numpy as np


class BaseIntentModel(ABC):
    """
    Abstract base class for all intent prediction models.
    Enforces a consistent interface across Logistic, XGBoost,
    Neural Net, and Ensemble models.
    """

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns binary predictions."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns probability of positive class."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass
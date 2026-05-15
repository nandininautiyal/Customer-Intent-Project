import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from loguru import logger
from models.base_model import BaseIntentModel


class IntentNet(nn.Module):
    """
    3-layer feedforward network with BatchNorm and Dropout.
    BatchNorm requires batch size > 1 — enforced via drop_last=True
    in the DataLoader.
    """

    def __init__(self, input_dim: int):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.block2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.block3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.output(x)


class NeuralIntentModel(BaseIntentModel):
    """
    PyTorch neural network with:
    - Weighted BCE loss for class imbalance
    - CosineAnnealingLR scheduler
    - Best model checkpointing
    - Automatic F1-optimal threshold tuning post-training
    - drop_last=True on train DataLoader to prevent
      single-sample batches crashing BatchNorm1d
    """

    def __init__(
        self,
        input_dim:  int,
        epochs:     int   = 60,
        lr:         float = 1e-3,
        batch_size: int   = 256
    ):
        self.input_dim  = input_dim
        self.epochs     = epochs
        self.lr         = lr
        self.batch_size = batch_size
        self.device     = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model             = IntentNet(input_dim).to(self.device)
        self.optimal_threshold = 0.5
        logger.info(f"[NeuralNet] Device: {self.device}")

    def _get_loader(
        self,
        X:        np.ndarray,
        y:        np.ndarray,
        shuffle:  bool = True,
        drop_last: bool = False
    ) -> DataLoader:
        """
        Builds a DataLoader from numpy arrays.

        drop_last=True on training loader prevents the last batch
        from having size 1, which crashes BatchNorm1d.
        drop_last=False on val/test loaders — we want all samples.
        """
        X_t = torch.FloatTensor(np.array(X)).to(self.device)
        y_t = torch.FloatTensor(np.array(y)).to(self.device)
        dataset = TensorDataset(X_t, y_t)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last
        )

    def _tune_threshold(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> float:
        """Finds the F1-optimal threshold on the validation set."""
        probas      = self._raw_probas(X_val)
        best_thresh = 0.5
        best_f1     = 0.0

        for thresh in np.arange(0.20, 0.81, 0.01):
            preds = (probas >= thresh).astype(int)
            score = f1_score(y_val, preds, zero_division=0)
            if score > best_f1:
                best_f1     = score
                best_thresh = thresh

        logger.info(
            f"[NeuralNet] Optimal threshold: {best_thresh:.2f} "
            f"→ F1: {best_f1:.4f}"
        )
        return float(best_thresh)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray,
        y_val:   np.ndarray
    ) -> None:
        y_train = np.array(y_train)
        y_val   = np.array(y_val)

        # Positive class weight from training distribution
        pos_weight = torch.tensor(
            [(y_train == 0).sum() / max((y_train == 1).sum(), 1)]
        ).to(self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-5
        )

        # drop_last=True on train loader — prevents batch size 1
        # hitting BatchNorm1d during OOF fold training
        train_loader = self._get_loader(
            X_train, y_train, shuffle=True, drop_last=True
        )
        val_loader = self._get_loader(
            X_val, y_val, shuffle=False, drop_last=False
        )

        best_val_loss = float("inf")
        best_state    = None

        for epoch in range(self.epochs):

            # --- Train ---
            self.model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                logits = self.model(X_batch).squeeze()
                # squeeze() on a batch of 1 returns a scalar — guard it
                if logits.dim() == 0:
                    logits = logits.unsqueeze(0)
                loss = criterion(logits, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )
                optimizer.step()

            scheduler.step()

            # --- Validate ---
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    logits = self.model(X_batch).squeeze()
                    if logits.dim() == 0:
                        logits = logits.unsqueeze(0)
                    val_loss += criterion(logits, y_batch).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {
                    k: v.clone()
                    for k, v in self.model.state_dict().items()
                }

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"[NeuralNet] Epoch {epoch+1}/{self.epochs} "
                    f"| Val Loss: {val_loss:.4f}"
                )

        # Restore best checkpoint
        self.model.load_state_dict(best_state)
        logger.info("[NeuralNet] Best checkpoint restored.")

        # Tune decision threshold on validation set
        self.optimal_threshold = self._tune_threshold(X_val, y_val)

    def _raw_probas(self, X: np.ndarray) -> np.ndarray:
        """Returns raw sigmoid probabilities without applying threshold."""
        self.model.eval()
        X_t = torch.FloatTensor(np.array(X)).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t).squeeze()
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self._raw_probas(X) >= self.optimal_threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._raw_probas(X)

    def save(self, path: str) -> None:
        torch.save(
            {
                "weights":   self.model.state_dict(),
                "threshold": self.optimal_threshold
            },
            path
        )
        logger.info(f"[NeuralNet] Saved to {path}")

    def load(self, path: str) -> None:
        data = torch.load(path, map_location=self.device)
        self.model.load_state_dict(data["weights"])
        self.optimal_threshold = data.get("threshold", 0.5)
        logger.info(
            f"[NeuralNet] Loaded. Threshold: {self.optimal_threshold:.2f}"
        )
import numpy as np
import joblib
from pathlib import Path
from loguru import logger


ACTIONS = [
    "show_social_proof",
    "show_discount",
    "show_urgency",
    "show_checkout_prompt",
    "show_exit_intent_offer",
    "show_seasonal_urgency",
    "show_returning_visitor_offer",
    "show_product_recommendation",
]

ARTIFACTS_DIR = Path("reports/artifacts")


class ContextualBandit:
    """
    LinUCB Contextual Bandit Recommender.

    Why LinUCB over pure rules:
    - Rules have hardcoded thresholds (exit_risk > 0.15) that were guessed
    - LinUCB LEARNS which action works best given session context
    - Upper Confidence Bound exploration means it tries new actions when uncertain
      and exploits known good actions when confident
    - Reward signal: purchase_probability is used as a proxy for conversion reward
      (higher probability session that received action X = action X was good)

    Architecture:
        For each action a, maintain:
            A_a  : (d x d) matrix — tracks feature covariance
            b_a  : (d,)   vector — tracks reward-weighted features
        At inference:
            theta_a = A_a^{-1} b_a     (estimated reward weights)
            UCB     = theta_a·x + alpha * sqrt(x·A_a^{-1}·x)
                      (expected reward + exploration bonus)
        Choose action with highest UCB.
    """

    def __init__(self, n_features: int = 6, alpha: float = 1.0):
        self.n_features = n_features
        self.alpha = alpha          # exploration coefficient
        self.actions = ACTIONS
        self.n_actions = len(ACTIONS)

        # One (A, b) pair per action
        self.A = [np.identity(n_features) for _ in range(self.n_actions)]
        self.b = [np.zeros(n_features)    for _ in range(self.n_actions)]

        # Track how many times each action was chosen (for reporting)
        self.action_counts = np.zeros(self.n_actions)
        self.total_reward  = np.zeros(self.n_actions)

    def _context_vector(self, session: dict) -> np.ndarray:
        """
        Extracts 6 normalized context features from session dict.
        These are the features the bandit uses to learn action preferences.
        """
        x = np.array([
            float(session.get("purchase_probability", 0.5)),
            float(session.get("ExitBounceRisk", 0.0)),
            float(session.get("PageValues", 0.0)) / 100.0,     # normalize
            float(session.get("ProductPageRatio", 0.0)),
            float(session.get("VisitorType", 0)) / 2.0,        # 0,1,2 → 0,0.5,1
            float(session.get("NearSpecialDay", 0)),
        ], dtype=np.float64)
        return x

    def select_action(self, session: dict) -> tuple[str, float, np.ndarray]:
        """
        Selects best action using LinUCB upper confidence bound.

        Returns:
            action_name : str
            ucb_score   : float (confidence score for this action)
            context_vec : np.ndarray (for update step)
        """
        x = self._context_vector(session)
        ucb_scores = []

        for a in range(self.n_actions):
            A_inv   = np.linalg.inv(self.A[a])
            theta_a = A_inv @ self.b[a]
            # Expected reward + exploration bonus
            expected = theta_a @ x
            bonus    = self.alpha * np.sqrt(x @ A_inv @ x)
            ucb_scores.append(expected + bonus)

        best_action_idx = int(np.argmax(ucb_scores))
        self.action_counts[best_action_idx] += 1

        return (
            self.actions[best_action_idx],
            float(ucb_scores[best_action_idx]),
            x
        )

    def update(self, action_name: str, context_vec: np.ndarray, reward: float) -> None:
        """
        Updates A and b matrices for the chosen action given observed reward.

        reward: float in [0, 1]
            In production: 1 if user purchased, 0 otherwise (from event tracking)
            At inference time: purchase_probability as proxy reward
        """
        a = self.actions.index(action_name)
        self.A[a] += np.outer(context_vec, context_vec)
        self.b[a] += reward * context_vec
        self.total_reward[a] += reward

    def get_action_stats(self) -> dict:
        """Returns summary of how often each action was chosen and avg reward."""
        stats = {}
        for i, action in enumerate(self.actions):
            count = int(self.action_counts[i])
            avg_reward = (
                float(self.total_reward[i] / count) if count > 0 else 0.0
            )
            stats[action] = {
                "times_chosen": count,
                "avg_reward": round(avg_reward, 4)
            }
        return stats

    def save(self, path: str) -> None:
        joblib.dump({
            "A": self.A,
            "b": self.b,
            "action_counts": self.action_counts,
            "total_reward": self.total_reward,
            "alpha": self.alpha,
            "n_features": self.n_features,
        }, path)
        logger.info(f"[Bandit] Saved to {path}")

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.A             = data["A"]
        self.b             = data["b"]
        self.action_counts = data["action_counts"]
        self.total_reward  = data["total_reward"]
        self.alpha         = data["alpha"]
        self.n_features    = data["n_features"]
        logger.info(f"[Bandit] Loaded from {path}")

    def warm_start(self, n_samples: int = 2000) -> None:
        """
        Simulates historical interactions to warm-start the bandit
        so it doesn't start completely cold on first deployment.

        Uses realistic session distributions derived from UCI dataset stats.
        """
        logger.info(f"[Bandit] Warm-starting with {n_samples} simulated interactions...")
        rng = np.random.default_rng(42)

        for _ in range(n_samples):
            # Sample a plausible session
            prob      = rng.beta(2, 8)          # skewed toward low intent (realistic)
            exit_risk = rng.beta(2, 5)
            page_val  = rng.exponential(10) / 100.0
            prod_ratio= rng.uniform(0, 1)
            visitor   = rng.choice([0, 0.5, 1], p=[0.4, 0.1, 0.5])
            special   = float(rng.random() < 0.1)

            session = {
                "purchase_probability": prob,
                "ExitBounceRisk": exit_risk,
                "PageValues": page_val * 100,
                "ProductPageRatio": prod_ratio,
                "VisitorType": visitor * 2,
                "NearSpecialDay": special,
            }

            action, _, ctx = self.select_action(session)

            # Simulated reward: higher intent + right action = better reward
            base_reward = prob
            # Actions appropriate to intent level get a small bonus
            if prob < 0.3 and action == "show_social_proof":
                base_reward += 0.1
            elif 0.3 <= prob < 0.55 and action in ("show_discount", "show_exit_intent_offer"):
                base_reward += 0.1
            elif 0.55 <= prob < 0.75 and action in ("show_urgency", "show_seasonal_urgency"):
                base_reward += 0.1
            elif prob >= 0.75 and action in ("show_checkout_prompt", "show_returning_visitor_offer"):
                base_reward += 0.1

            self.update(action, ctx, min(base_reward, 1.0))

        logger.info("[Bandit] Warm-start complete.")
        logger.info(f"[Bandit] Action distribution: {self.get_action_stats()}")
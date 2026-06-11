import numpy as np
import joblib
from pathlib import Path
from loguru import logger

from pipeline.segments import SEGMENT_ALLOWED_ACTIONS


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
    LinUCB Contextual Bandit Recommender with segment-constrained action selection.

    Why LinUCB over pure rules:
    - Rules have hardcoded thresholds (exit_risk > 0.15) that were guessed
    - LinUCB LEARNS which action works best given session context
    - Upper Confidence Bound exploration means it tries new actions when uncertain
      and exploits known good actions when confident
    - Reward signal: purchase_probability is used as a proxy for conversion reward

    Segment constraints (new):
    - Hard business rules define which actions are valid per segment
    - The bandit scores only the allowed subset, then picks the best
    - This prevents category errors (e.g. exit-intent for high-intent visitors)
      while still learning fine-grained preferences within each segment

    Architecture:
        For each action a, maintain:
            A_a  : (d x d) matrix — tracks feature covariance
            b_a  : (d,)   vector — tracks reward-weighted features
        At inference:
            theta_a = A_a^{-1} b_a     (estimated reward weights)
            UCB     = theta_a·x + alpha * sqrt(x·A_a^{-1}·x)
                      (expected reward + exploration bonus)
        Choose action with highest UCB *within the segment's allowed set*.
    """

    def __init__(self, n_features: int = 6, alpha: float = 1.0):
        self.n_features = n_features
        self.alpha = alpha
        self.actions = ACTIONS
        self.n_actions = len(ACTIONS)

        # One (A, b) pair per action — maintained across all segments so that
        # the bandit accumulates global knowledge about each action.
        self.A = [np.identity(n_features) for _ in range(self.n_actions)]
        self.b = [np.zeros(n_features)    for _ in range(self.n_actions)]

        self.action_counts = np.zeros(self.n_actions)
        self.total_reward  = np.zeros(self.n_actions)

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------

    def _context_vector(self, session: dict) -> np.ndarray:
        """
        Extracts 6 normalized context features from session dict.
        These are the features the bandit uses to learn action preferences.
        """
        x = np.array([
            float(session.get("purchase_probability", 0.5)),
            float(session.get("ExitBounceRisk", 0.0)),
            float(session.get("PageValues", 0.0)) / 100.0,
            float(session.get("ProductPageRatio", 0.0)),
            float(session.get("VisitorType", 0)) / 2.0,      # 0,1,2 → 0,0.5,1
            float(session.get("NearSpecialDay", 0)),
        ], dtype=np.float64)
        return x

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def select_action(
        self,
        session: dict,
        allowed_actions: list[str] | None = None,
    ) -> tuple[str, float, np.ndarray]:
        """
        Selects the best action using LinUCB upper confidence bound,
        restricted to *allowed_actions* when provided.

        Args:
            session:         Full session feature dict (must include
                             'purchase_probability').
            allowed_actions: Subset of ACTIONS to score.  When None, all
                             actions are considered (used during warm-start
                             or when called without segment context).

        Returns:
            action_name : str
            ucb_score   : float  — normalised to [0, 1] for clean API output
            context_vec : np.ndarray — returned for the subsequent update call
        """
        x = self._context_vector(session)

        candidate_indices = (
            [self.actions.index(a) for a in allowed_actions if a in self.actions]
            if allowed_actions is not None
            else list(range(self.n_actions))
        )

        if not candidate_indices:
            # Safety fallback: if constraint produces empty set, use all actions
            logger.warning("select_action received an empty allowed_actions list; using all actions.")
            candidate_indices = list(range(self.n_actions))

        ucb_scores = []
        for a in candidate_indices:
            A_inv    = np.linalg.inv(self.A[a])
            theta_a  = A_inv @ self.b[a]
            expected = theta_a @ x
            bonus    = self.alpha * np.sqrt(x @ A_inv @ x)
            ucb_scores.append(expected + bonus)

        best_local_idx   = int(np.argmax(ucb_scores))
        best_action_idx  = candidate_indices[best_local_idx]
        raw_ucb          = float(ucb_scores[best_local_idx])

        self.action_counts[best_action_idx] += 1

        # Normalise UCB score to [0, 1] using sigmoid so the API always
        # returns a bounded confidence value.
        normalised_score = float(1.0 / (1.0 + np.exp(-raw_ucb)))

        return (
            self.actions[best_action_idx],
            normalised_score,
            x,
        )

    # ------------------------------------------------------------------
    # Online update
    # ------------------------------------------------------------------

    def update(self, action_name: str, context_vec: np.ndarray, reward: float) -> None:
        """
        Updates A and b matrices for the chosen action given observed reward.

        reward: float in [0, 1]
            Production: 1 if user purchased, 0 otherwise (event tracking)
            Proxy mode: purchase_probability at inference time
        """
        a = self.actions.index(action_name)
        self.A[a] += np.outer(context_vec, context_vec)
        self.b[a] += reward * context_vec
        self.total_reward[a] += reward

    # ------------------------------------------------------------------
    # Warm-start
    # ------------------------------------------------------------------

    def warm_start(self, n_samples: int = 2000) -> None:
        """
        Simulates historical interactions to warm-start the bandit.

        Key fix vs original: reward shaping now uses a ×3 multiplier for
        correct-segment actions, creating a much stronger prior.  The bandit
        still explores, but it starts from a sensible baseline rather than
        a uniform prior that is easily dominated by the exploration bonus.

        Segment constraints are enforced during warm-start so the bandit
        builds per-segment A/b matrices from the start.
        """
        from pipeline.segments import assign_segment, SEGMENT_ALLOWED_ACTIONS

        logger.info(f"[Bandit] Warm-starting with {n_samples} simulated interactions...")
        rng = np.random.default_rng(42)

        # Rough segment → reward shaping map used during simulation
        _SEGMENT_PREFERRED: dict[str, list[str]] = {
            "Cold":    ["show_social_proof", "show_product_recommendation"],
            "Warm":    ["show_discount", "show_exit_intent_offer"],
            "Hot":     ["show_urgency", "show_seasonal_urgency"],
            "Convert": ["show_checkout_prompt", "show_returning_visitor_offer"],
        }

        for _ in range(n_samples):
            prob       = rng.beta(2, 8)
            exit_risk  = rng.beta(2, 5)
            page_val   = rng.exponential(10) / 100.0
            prod_ratio = rng.uniform(0, 1)
            visitor    = rng.choice([0, 0.5, 1], p=[0.4, 0.1, 0.5])
            special    = float(rng.random() < 0.1)

            session = {
                "purchase_probability": prob,
                "ExitBounceRisk":       exit_risk,
                "PageValues":           page_val * 100,
                "ProductPageRatio":     prod_ratio,
                "VisitorType":          visitor * 2,
                "NearSpecialDay":       special,
            }

            segment         = assign_segment(prob)
            allowed         = SEGMENT_ALLOWED_ACTIONS[segment]
            action, _, ctx  = self.select_action(session, allowed_actions=allowed)

            # Reward: base = probability; preferred actions get a ×3 bonus
            # so the warm-start creates a meaningful prior, not a noise signal
            base_reward = prob
            preferred   = _SEGMENT_PREFERRED.get(segment, [])
            if action in preferred:
                base_reward = min(prob * 3.0, 1.0)

            self.update(action, ctx, base_reward)

        logger.info("[Bandit] Warm-start complete.")
        logger.info(f"[Bandit] Action distribution: {self.get_action_stats()}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        joblib.dump({
            "A":             self.A,
            "b":             self.b,
            "action_counts": self.action_counts,
            "total_reward":  self.total_reward,
            "alpha":         self.alpha,
            "n_features":    self.n_features,
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

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_action_stats(self) -> dict:
        """Returns a summary of how often each action was chosen and avg reward."""
        stats = {}
        for i, action in enumerate(self.actions):
            count = int(self.action_counts[i])
            avg_reward = (
                float(self.total_reward[i] / count) if count > 0 else 0.0
            )
            stats[action] = {
                "times_chosen": count,
                "avg_reward":   round(avg_reward, 4),
            }
        return stats
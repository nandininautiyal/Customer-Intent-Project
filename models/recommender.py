import numpy as np
from pathlib import Path
from loguru import logger
from pipeline.segments import assign_segment, SEGMENT_STRATEGY


ACTION_LABELS = {
    "show_social_proof":             "Show Reviews & Trust Badges",
    "show_discount":                 "Offer Discount / Free Shipping",
    "show_urgency":                  "Show Low Stock / Popularity Signal",
    "show_checkout_prompt":          "Direct Checkout CTA",
    "show_exit_intent_offer":        "Exit-Intent Popup with Offer",
    "show_seasonal_urgency":         "Seasonal / Special Day Offer",
    "show_returning_visitor_offer":  "Returning Visitor Loyalty Reward",
    "show_product_recommendation":   "Personalised Product Recommendations",
}

ACTION_REASONS = {
    "show_social_proof":             "Low intent visitor — build credibility with ratings and reviews.",
    "show_discount":                 "Mid-funnel visitor — a discount nudge can tip the decision.",
    "show_urgency":                  "High intent visitor — urgency signal can close the sale.",
    "show_checkout_prompt":          "Very high intent — remove friction, surface checkout immediately.",
    "show_exit_intent_offer":        "High exit risk detected — intercept with an exit-intent offer.",
    "show_seasonal_urgency":         "Session near a special day — highlight time-limited seasonal deal.",
    "show_returning_visitor_offer":  "Returning visitor with strong signals — reward loyalty.",
    "show_product_recommendation":   "Engaged but unfocused — guide toward relevant products.",
}


class IntentRecommender:
    """
    Wraps the LinUCB ContextualBandit and provides a clean interface
    for the API layer.

    At inference:
      1. Bandit selects best action given session context
      2. Bandit updates itself using purchase_probability as proxy reward
      3. Returns action + human-readable label + reason + segment
    """

    def __init__(self, bandit=None):
        self.bandit = bandit

    def recommend(self, probability: float, session_features: dict) -> dict:
        segment = assign_segment(probability)
        base_strategy = SEGMENT_STRATEGY[segment]

        session_context = {**session_features, "purchase_probability": probability}

        if self.bandit is not None:
            action, ucb_score, ctx_vec = self.bandit.select_action(session_context)
            # Online update: use probability as proxy reward signal
            self.bandit.update(action, ctx_vec, reward=probability)
        else:
            # Fallback to rule-based if bandit not loaded
            action = base_strategy["action"]
            ucb_score = 0.0

        return {
            "segment":              segment,
            "recommended_action":   action,
            "action_label":         ACTION_LABELS.get(action, action),
            "reason":               ACTION_REASONS.get(action, ""),
            "urgency":              base_strategy["urgency"],
            "message":              base_strategy["message"],
            "confidence_score":     round(float(ucb_score), 4),
        }

    def batch_recommend(self, probabilities: np.ndarray,
                        session_features_list: list) -> list:
        return [
            self.recommend(prob, features)
            for prob, features in zip(probabilities, session_features_list)
        ]
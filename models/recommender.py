import numpy as np
from pipeline.segments import assign_segment, SEGMENT_STRATEGY
from loguru import logger


class IntentRecommender:
    """
    Rule-based recommender that maps purchase probability + session
    features to a concrete marketing intervention.

    Design philosophy:
    - Segment is determined by probability score
    - Action is refined by session context (exit risk, page value, visitor type)
    - Reason is generated to be human-readable for business teams
    """

    def recommend(self, probability: float, session_features: dict) -> dict:
        """
        Args:
            probability: float between 0 and 1 from ensemble model
            session_features: dict of raw/engineered session values

        Returns:
            dict with segment, action, reason, urgency
        """
        segment = assign_segment(probability)
        base_strategy = SEGMENT_STRATEGY[segment].copy()

        # Refine based on session context
        action, reason = self._refine(segment, probability, session_features, base_strategy["action"])

        return {
            "segment": segment,
            "recommended_action": action,
            "reason": reason,
            "urgency": base_strategy["urgency"],
            "message": base_strategy["message"],
        }

    def _refine(self, segment: str, prob: float,
                features: dict, default_action: str) -> tuple[str, str]:
        """
        Contextual refinement rules on top of base segment strategy.
        Each rule is independently explainable.
        """

        exit_risk = features.get("ExitBounceRisk", 0)
        page_value = features.get("PageValues", 0)
        product_ratio = features.get("ProductPageRatio", 0)
        visitor_type = features.get("VisitorType", 0)  # 0=new, 1=other, 2=returning
        avg_time = features.get("AvgTimePerPage", 0)
        near_special = features.get("NearSpecialDay", 0)

        # --- Rule overrides (priority order) ---

        # High exit risk on a warm/hot session = intervene NOW with discount
        if exit_risk > 0.15 and segment in ("Warm", "Hot"):
            return (
                "show_exit_intent_offer",
                "High exit risk detected — trigger exit-intent popup with discount."
            )

        # Near a special day = seasonal urgency works best
        if near_special and segment in ("Warm", "Hot", "Convert"):
            return (
                "show_seasonal_urgency",
                "Session near a special day — highlight limited-time seasonal offer."
            )

        # High page value + returning visitor = they've been here before, push checkout
        if page_value > 20 and visitor_type == 2 and segment in ("Hot", "Convert"):
            return (
                "show_returning_visitor_offer",
                "Returning visitor with high page value — offer loyalty reward or saved cart reminder."
            )

        # Deeply engaged but low product ratio = browsing broadly, needs direction
        if avg_time > 5 and product_ratio < 0.3 and segment == "Warm":
            return (
                "show_product_recommendation",
                "High engagement but low product focus — surface personalized product recommendations."
            )

        # New visitor, cold segment = build trust first
        if visitor_type == 0 and segment == "Cold":
            return (
                "show_social_proof",
                "New visitor with low intent — show trust signals and top-rated products."
            )

        # Default to base strategy
        return default_action, f"Standard {segment} intervention based on purchase probability {prob:.2f}."

    def batch_recommend(self, probabilities: np.ndarray,
                        session_features_list: list[dict]) -> list[dict]:
        """Runs recommend() across a batch of sessions."""
        return [
            self.recommend(prob, features)
            for prob, features in zip(probabilities, session_features_list)
        ]
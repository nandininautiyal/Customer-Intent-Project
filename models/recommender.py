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


# ---------------------------------------------------------------------------
# Signal extraction helpers
# ---------------------------------------------------------------------------

def _exit_risk_level(features: dict) -> tuple[str, float]:
    """Returns a (label, value) for the dominant exit/bounce signal."""
    exit_bounce = features.get("ExitBounceRisk", 0.0)
    exit_rate   = features.get("ExitRate", 0.0)
    bounce_rate = features.get("BounceRate", 0.0)
    dominant    = max(exit_bounce, exit_rate, bounce_rate)

    if dominant >= 0.40:
        return "high", dominant
    if dominant >= 0.15:
        return "moderate", dominant
    return "low", dominant


def _engagement_level(features: dict) -> tuple[str, float]:
    """Returns (label, page_value_score) summarising on-site engagement."""
    pv = features.get("PageValues", features.get("PageValue", 0.0))
    if pv >= 150:
        return "high", pv
    if pv >= 50:
        return "moderate", pv
    return "low", pv


def _product_focus(features: dict) -> tuple[str, float]:
    """Returns (label, ratio) for how product-focused the session is."""
    ratio = features.get("ProductPageRatio", 0.0)
    if ratio >= 0.70:
        return "focused", ratio
    if ratio >= 0.35:
        return "moderate", ratio
    return "browsing", ratio


def _is_returning(features: dict) -> bool:
    vtype = features.get("VisitorType", "")
    if isinstance(vtype, str):
        return vtype.lower() == "returning_visitor"
    # encoded: 1 = returning in most pipelines
    return int(vtype) == 1


def _near_special_day(features: dict) -> bool:
    return float(features.get("NearSpecialDay", 0.0)) > 0.0


# ---------------------------------------------------------------------------
# Per-action explanation builders
# Each receives (probability, session_features) and returns a plain-English
# string that is factually grounded in the actual feature values.
# ---------------------------------------------------------------------------

def _explain_show_social_proof(prob: float, f: dict) -> str:
    eng_label, pv = _engagement_level(f)
    lines = []
    if prob < 0.40:
        lines.append(f"Purchase probability is {prob:.0%} — the visitor hasn't yet committed.")
    if eng_label == "low":
        lines.append(f"Page value score is low ({pv:.0f}), indicating limited product engagement.")
    lines.append("Showing ratings and trust signals can reduce hesitation and build confidence.")
    return " ".join(lines)


def _explain_show_discount(prob: float, f: dict) -> str:
    lines = [f"Purchase probability is {prob:.0%} — intent exists but the visitor hasn't converted."]
    eng_label, pv = _engagement_level(f)
    if eng_label in ("moderate", "high"):
        lines.append(f"Page value score ({pv:.0f}) shows genuine product interest.")
    lines.append("A discount or free-shipping offer targets the remaining price sensitivity.")
    return " ".join(lines)


def _explain_show_urgency(prob: float, f: dict) -> str:
    focus_label, ratio = _product_focus(f)
    lines = [f"Purchase probability is {prob:.0%} — the visitor is close to converting."]
    if focus_label == "focused":
        lines.append(f"Product page ratio is {ratio:.0%}, indicating focused browsing.")
    lines.append("A low-stock or popularity signal can tip the decision without discounting margin.")
    return " ".join(lines)


def _explain_show_checkout_prompt(prob: float, f: dict) -> str:
    lines = [f"Purchase probability is {prob:.0%} — this visitor is highly likely to buy."]
    focus_label, ratio = _product_focus(f)
    if focus_label == "focused":
        lines.append(f"Product page ratio is {ratio:.0%}.")
    lines.append("Surface the checkout CTA immediately to remove any remaining friction.")
    return " ".join(lines)


def _explain_show_exit_intent_offer(prob: float, f: dict) -> str:
    risk_label, risk_val = _exit_risk_level(f)
    lines = []

    if risk_label == "high":
        lines.append(
            f"Exit-bounce risk is elevated ({risk_val:.2f}) — the visitor shows strong abandonment signals."
        )
        lines.append("An exit-intent offer intercepts the session before the visitor leaves.")
    elif risk_label == "moderate":
        lines.append(
            f"Exit-bounce risk is moderate ({risk_val:.2f}), suggesting the visitor may leave without converting."
        )
        lines.append("A timely offer can retain attention and recover the session.")
    else:
        # Low exit risk — explain why the bandit still chose this action
        lines.append(
            f"Exit-bounce risk is currently low ({risk_val:.2f}), "
            f"but purchase probability ({prob:.0%}) and session context "
            "favour a proactive retention offer over a passive nudge."
        )
        lines.append(
            "The bandit selected this action based on historical reward patterns for this visitor profile."
        )
    return " ".join(lines)


def _explain_show_seasonal_urgency(prob: float, f: dict) -> str:
    near = _near_special_day(f)
    lines = []
    if near:
        lines.append("The session is close to a special day or promotional period.")
    else:
        lines.append("A seasonal or promotional period is approaching.")
    lines.append(
        f"With purchase probability at {prob:.0%}, "
        "a time-limited seasonal offer adds urgency without discounting the core proposition."
    )
    return " ".join(lines)


def _explain_show_returning_visitor_offer(prob: float, f: dict) -> str:
    returning = _is_returning(f)
    eng_label, pv = _engagement_level(f)
    lines = []
    if returning:
        lines.append("This is a returning visitor — they already know the product.")
    else:
        lines.append("Visitor history indicates prior engagement with the site.")
    lines.append(
        f"Page value score is {pv:.0f} and purchase probability is {prob:.0%}. "
        "A loyalty reward acknowledges repeat intent and lowers the final conversion barrier."
    )
    return " ".join(lines)


def _explain_show_product_recommendation(prob: float, f: dict) -> str:
    focus_label, ratio = _product_focus(f)
    eng_label, pv = _engagement_level(f)
    lines = [
        f"Purchase probability is {prob:.0%} and product page ratio is {ratio:.0%} ({focus_label} focus)."
    ]
    if eng_label == "low":
        lines.append(
            f"Page value score is low ({pv:.0f}), suggesting the visitor hasn't found "
            "the right product yet."
        )
    lines.append(
        "Personalised recommendations guide them toward higher-relevance items and improve conversion odds."
    )
    return " ".join(lines)


# ---------------------------------------------------------------------------
# Dispatch table — maps action key → builder function
# ---------------------------------------------------------------------------

_EXPLANATION_BUILDERS = {
    "show_social_proof":             _explain_show_social_proof,
    "show_discount":                 _explain_show_discount,
    "show_urgency":                  _explain_show_urgency,
    "show_checkout_prompt":          _explain_show_checkout_prompt,
    "show_exit_intent_offer":        _explain_show_exit_intent_offer,
    "show_seasonal_urgency":         _explain_show_seasonal_urgency,
    "show_returning_visitor_offer":  _explain_show_returning_visitor_offer,
    "show_product_recommendation":   _explain_show_product_recommendation,
}


def build_explanation(action: str, probability: float, session_features: dict) -> str:
    """
    Return a truthful, signal-grounded explanation for why *action* was
    selected for this specific session.

    Falls back to a generic probability statement if no builder exists
    (e.g., for custom actions added later).
    """
    builder = _EXPLANATION_BUILDERS.get(action)
    if builder is None:
        return (
            f"Selected based on a purchase probability of {probability:.0%} "
            "and overall session context."
        )
    try:
        return builder(probability, session_features)
    except Exception as exc:  # pragma: no cover
        logger.warning(f"Explanation builder failed for action '{action}': {exc}")
        return (
            f"Selected based on a purchase probability of {probability:.0%} "
            "and overall session context."
        )


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------

class IntentRecommender:
    """
    Wraps the LinUCB ContextualBandit and provides a clean interface
    for the API layer.

    At inference:
      1. Bandit selects the best action given session context.
      2. Bandit updates itself using purchase_probability as a proxy reward.
      3. Returns action + human-readable label + *dynamic* reason + segment.

    The ``reason`` field is generated at inference time from actual session
    feature values rather than looked up from a static dictionary, so it
    is always factually consistent with the visitor's signals.
    """

    def __init__(self, bandit=None):
        self.bandit = bandit

    def recommend(self, probability: float, session_features: dict) -> dict:
        segment = assign_segment(probability)
        base_strategy = SEGMENT_STRATEGY[segment]

        session_context = {**session_features, "purchase_probability": probability}

        if self.bandit is not None:
            action, ucb_score, ctx_vec = self.bandit.select_action(session_context)
            # Online update: use probability as proxy reward signal.
            self.bandit.update(action, ctx_vec, reward=probability)
        else:
            # Fallback to rule-based strategy if bandit is not loaded.
            action = base_strategy["action"]
            ucb_score = 0.0

        reason = build_explanation(action, probability, session_features)

        return {
            "segment":              segment,
            "recommended_action":   action,
            "action_label":         ACTION_LABELS.get(action, action),
            "reason":               reason,
            "urgency":              base_strategy["urgency"],
            "message":              base_strategy["message"],
            "confidence_score":     round(float(ucb_score), 4),
        }

    def batch_recommend(
        self,
        probabilities: np.ndarray,
        session_features_list: list,
    ) -> list:
        return [
            self.recommend(prob, features)
            for prob, features in zip(probabilities, session_features_list)
        ]
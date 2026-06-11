import numpy as np
import pandas as pd
from loguru import logger


# Probability thresholds defining each segment
SEGMENT_THRESHOLDS = {
    "Cold":    (0.00, 0.30),
    "Warm":    (0.30, 0.55),
    "Hot":     (0.55, 0.75),
    "Convert": (0.75, 1.00),
}

# What each segment should receive as its *default* fallback action
# (used when bandit is unavailable or during first-cold-start)
SEGMENT_STRATEGY = {
    "Cold":    {
        "action": "show_social_proof",
        "message": "Show reviews, ratings, and trust badges to build credibility.",
        "urgency": "low"
    },
    "Warm":    {
        "action": "show_discount",
        "message": "Offer a time-limited discount or free shipping nudge.",
        "urgency": "medium"
    },
    "Hot":     {
        "action": "show_urgency",
        "message": "Show low stock warning or 'X people viewing this' signal.",
        "urgency": "high"
    },
    "Convert": {
        "action": "show_checkout_prompt",
        "message": "Surface a direct CTA — streamline path to checkout immediately.",
        "urgency": "critical"
    },
}

# ---------------------------------------------------------------------------
# Segment action constraints
# ---------------------------------------------------------------------------
# The bandit selects from ONLY this subset of actions for each segment.
# This enforces business logic as a hard constraint while still allowing
# the bandit to learn which action within the set performs best for a
# given context profile.
#
# Design rationale per segment:
#   Cold    — visitor needs credibility signals; discount is also valid
#             because a price incentive can move a cold visitor faster than
#             trust-building alone.  Exit-intent is appropriate if the visitor
#             looks like they are about to leave.
#
#   Warm    — mid-funnel; discount, urgency, and product recommendations are
#             all legitimate nudges.  Exit-intent is included because warm
#             visitors are the most common abandoners.
#
#   Hot     — close to converting; urgency and checkout prompts are primary.
#             Returning-visitor reward is included because returning hot
#             visitors respond well to loyalty recognition.
#             Exit-intent is excluded — these visitors have high intent and
#             triggering an exit popup is premature and can feel intrusive.
#
#   Convert — highest-intent; only checkout prompt and returning-visitor
#             reward make sense.  Any discount or urgency signal at this
#             stage is redundant and may erode margin.
# ---------------------------------------------------------------------------
SEGMENT_ALLOWED_ACTIONS: dict[str, list[str]] = {
    "Cold": [
        "show_social_proof",
        "show_discount",
        "show_exit_intent_offer",
        "show_product_recommendation",
    ],
    "Warm": [
        "show_discount",
        "show_urgency",
        "show_exit_intent_offer",
        "show_product_recommendation",
        "show_seasonal_urgency",
    ],
    "Hot": [
        "show_urgency",
        "show_checkout_prompt",
        "show_returning_visitor_offer",
        "show_seasonal_urgency",
        "show_product_recommendation",
    ],
    "Convert": [
        "show_checkout_prompt",
        "show_returning_visitor_offer",
        "show_seasonal_urgency",
    ],
}


def assign_segment(probability: float) -> str:
    """Maps a purchase probability to a named segment."""
    for segment, (low, high) in SEGMENT_THRESHOLDS.items():
        if low <= probability < high:
            return segment
    return "Convert"  # catch 1.0


def get_allowed_actions(segment: str) -> list[str]:
    """Returns the list of actions the bandit may select for this segment."""
    return SEGMENT_ALLOWED_ACTIONS[segment]


def get_segment_distribution(probas: np.ndarray) -> dict:
    """
    Given an array of probabilities, returns segment counts and percentages.
    Useful for reporting and business dashboards.
    """
    segments = [assign_segment(p) for p in probas]
    total = len(segments)
    dist = {}
    for seg in SEGMENT_THRESHOLDS:
        count = segments.count(seg)
        dist[seg] = {
            "count": count,
            "percentage": round(count / total * 100, 1)
        }
    logger.info(f"Segment distribution: { {k: v['percentage'] for k, v in dist.items()} }")
    return dist
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

# What each segment should receive
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


def assign_segment(probability: float) -> str:
    """Maps a purchase probability to a named segment."""
    for segment, (low, high) in SEGMENT_THRESHOLDS.items():
        if low <= probability < high:
            return segment
    return "Convert"  # catch 1.0


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
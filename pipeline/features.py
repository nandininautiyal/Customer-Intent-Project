import pandas as pd
import numpy as np
from loguru import logger


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds domain-specific features to improve model signal.
    All features are interpretable and business-meaningful.
    """
    df = df.copy()

    # Total pages visited across all categories
    df["TotalPages"] = (
        df["Administrative"] + df["Informational"] + df["ProductRelated"]
    )

    # Total time spent on site
    df["TotalDuration"] = (
        df["Administrative_Duration"] +
        df["Informational_Duration"] +
        df["ProductRelated_Duration"]
    )

    # Ratio of product pages to total pages (purchase intent signal)
    df["ProductPageRatio"] = df["ProductRelated"] / (df["TotalPages"] + 1e-6)

    # Average time per page (engagement depth)
    df["AvgTimePerPage"] = df["TotalDuration"] / (df["TotalPages"] + 1e-6)

    # High page value flag 
    pv_threshold = df["PageValues"].quantile(0.75)
    df["HighPageValue"] = (df["PageValues"] >= pv_threshold).astype(int)

    # Near special day flag
    df["NearSpecialDay"] = (df["SpecialDay"] > 0).astype(int)

    # Bounce + Exit combined risk score
    df["ExitBounceRisk"] = df["BounceRates"] + df["ExitRates"]

    logger.info(f"Engineered 7 new features. Total columns: {df.shape[1]}")
    return df
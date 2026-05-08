import pandas as pd
from pathlib import Path
from loguru import logger


RAW_DATA_PATH = Path("data/raw/online_shoppers_intention.csv")


def load_raw_data() -> pd.DataFrame:
    """
    Loads the UCI Online Shoppers Intention dataset from disk.
    Validates schema and logs basic statistics.
    """
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {RAW_DATA_PATH}. "
            "Download from: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset"
        )

    df = pd.read_csv(RAW_DATA_PATH)
    logger.info(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    _validate_schema(df)
    _log_class_balance(df)

    return df


def _validate_schema(df: pd.DataFrame) -> None:
    """Ensures all expected columns are present."""
    expected_columns = [
        "Administrative", "Administrative_Duration",
        "Informational", "Informational_Duration",
        "ProductRelated", "ProductRelated_Duration",
        "BounceRates", "ExitRates", "PageValues",
        "SpecialDay", "Month", "OperatingSystems",
        "Browser", "Region", "TrafficType",
        "VisitorType", "Weekend", "Revenue"
    ]
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    logger.info("Schema validation passed.")


def _log_class_balance(df: pd.DataFrame) -> None:
    """Logs class distribution of target variable."""
    counts = df["Revenue"].value_counts()
    pct = df["Revenue"].value_counts(normalize=True) * 100
    logger.info(f"Class balance — True (purchase): {pct[True]:.1f}% | False: {pct[False]:.1f}%")
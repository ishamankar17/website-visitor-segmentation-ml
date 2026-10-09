"""Load, validate and clean the UCI Online Shoppers Purchasing Intention dataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/online_shoppers_intention.csv")
PROCESSED_PATH = Path("data/processed/visitors_cleaned.csv")

# The UCI file spells June as "June" while every other month is a 3-letter code.
MONTH_FIX = {"June": "Jun"}
MONTH_NUM = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}

COUNT_DURATION_COLS = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
]
RATE_COLS = ["BounceRates", "ExitRates"]
REQUIRED_COLS = COUNT_DURATION_COLS + RATE_COLS + [
    "PageValues", "SpecialDay", "Month", "OperatingSystems", "Browser",
    "Region", "TrafficType", "VisitorType", "Weekend", "Revenue",
]


def load_raw(path: str | Path = RAW_PATH) -> pd.DataFrame:
    """Read the raw CSV."""
    return pd.read_csv(path)


def validate_schema(df: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw dataframe.

    Steps
    -----
    1. Validate schema and strip column names.
    2. Remove exact duplicate sessions.
    3. Impute missing values (median for numeric, mode for categorical).
    4. Standardise month names and add a numeric month.
    5. Clip impossible negative values to zero and rates to [0, 1].
    6. Convert booleans to 0/1 integers.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    validate_schema(df)

    df = df.drop_duplicates()

    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0])

    df["Month"] = df["Month"].str.strip().replace(MONTH_FIX)
    df["Month_Num"] = df["Month"].map(MONTH_NUM)
    df["VisitorType"] = df["VisitorType"].str.strip()

    non_negative = COUNT_DURATION_COLS + ["PageValues"]
    df[non_negative] = df[non_negative].clip(lower=0)
    df[RATE_COLS] = df[RATE_COLS].clip(lower=0, upper=1)

    for col in ("Weekend", "Revenue"):
        df[col] = df[col].astype(int)

    return df.reset_index(drop=True)


def save_processed(df: pd.DataFrame, path: str | Path = PROCESSED_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def run(raw_path: str | Path = RAW_PATH, out_path: str | Path = PROCESSED_PATH) -> pd.DataFrame:
    """Full preprocessing step: raw CSV -> cleaned CSV."""
    cleaned = clean_data(load_raw(raw_path))
    save_processed(cleaned, out_path)
    return cleaned


if __name__ == "__main__":
    out = run()
    print(f"Saved {len(out):,} cleaned rows to {PROCESSED_PATH}")

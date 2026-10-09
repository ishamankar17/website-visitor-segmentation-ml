"""Behavioural feature engineering for visitor segmentation."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Raw columns needed to build the features (used by the Streamlit app form).
INPUT_COLS = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "VisitorType",
]

# Features fed to K-Means. `Revenue` is deliberately excluded: it is the outcome
# we want to *explain* the segments with, not an input to them.
CLUSTER_FEATURES = [
    "log_TotalPages",
    "log_TotalDuration",
    "log_AvgTimePerPage",
    "ProductPageShare",
    "BounceRates",
    "ExitRates",
    "log_PageValues",
    "IsNewVisitor",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add behavioural features to a (cleaned) dataframe and return a copy."""
    missing = set(INPUT_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for feature engineering: {sorted(missing)}")

    out = df.copy()

    out["TotalPages"] = out["Administrative"] + out["Informational"] + out["ProductRelated"]
    out["TotalDuration"] = (
        out["Administrative_Duration"]
        + out["Informational_Duration"]
        + out["ProductRelated_Duration"]
    )

    pages = out["TotalPages"].replace(0, np.nan)
    out["AvgTimePerPage"] = (out["TotalDuration"] / pages).fillna(0)
    out["ProductPageShare"] = (out["ProductRelated"] / pages).fillna(0)
    out["ProductTimeShare"] = (
        out["ProductRelated_Duration"] / out["TotalDuration"].replace(0, np.nan)
    ).fillna(0)

    out["IsNewVisitor"] = (out["VisitorType"] == "New_Visitor").astype(int)
    out["IsReturning"] = (out["VisitorType"] == "Returning_Visitor").astype(int)

    # log1p tames the heavy right skew of counts / durations / page values.
    for col in ("TotalPages", "TotalDuration", "AvgTimePerPage", "PageValues"):
        out[f"log_{col}"] = np.log1p(out[col])

    return out


def get_cluster_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features (if needed) and return only the clustering columns."""
    if not set(CLUSTER_FEATURES).issubset(df.columns):
        df = add_features(df)
    return df[CLUSTER_FEATURES].copy()

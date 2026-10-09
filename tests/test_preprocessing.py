import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import REQUIRED_COLS, clean_data
from src.feature_engineering import CLUSTER_FEATURES, add_features, get_cluster_matrix


def make_raw(n: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "Administrative": rng.integers(0, 5, n),
            "Administrative_Duration": rng.random(n) * 100,
            "Informational": rng.integers(0, 3, n),
            "Informational_Duration": rng.random(n) * 50,
            "ProductRelated": rng.integers(1, 40, n),
            "ProductRelated_Duration": rng.random(n) * 1000,
            "BounceRates": rng.random(n) * 0.2,
            "ExitRates": rng.random(n) * 0.2,
            "PageValues": rng.random(n) * 20,
            "SpecialDay": 0.0,
            "Month": ["June", "Feb", "May", "Nov", "Dec", "Mar"][:n],
            "OperatingSystems": 1,
            "Browser": 1,
            "Region": 1,
            "TrafficType": 1,
            "VisitorType": ["Returning_Visitor", "New_Visitor", "Other"] * (n // 3),
            "Weekend": [True, False] * (n // 2),
            "Revenue": [False, True] * (n // 2),
        }
    )
    return df[REQUIRED_COLS]


def test_clean_removes_duplicates():
    raw = make_raw()
    dup = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    assert len(clean_data(dup)) == len(raw)


def test_clean_standardises_month_and_types():
    out = clean_data(make_raw())
    assert "June" not in out["Month"].values
    assert out.loc[0, "Month"] == "Jun" and out.loc[0, "Month_Num"] == 6
    assert out["Revenue"].isin([0, 1]).all() and out["Weekend"].isin([0, 1]).all()


def test_clean_imputes_missing_and_clips_negatives():
    raw = make_raw()
    raw.loc[0, "ProductRelated_Duration"] = np.nan
    raw.loc[1, "Administrative_Duration"] = -5
    out = clean_data(raw)
    assert out.isna().sum().sum() == 0
    assert (out["Administrative_Duration"] >= 0).all()


def test_clean_rejects_missing_columns():
    with pytest.raises(ValueError):
        clean_data(make_raw().drop(columns=["Revenue"]))


def test_feature_engineering_values():
    df = pd.DataFrame([{
        "Administrative": 2, "Administrative_Duration": 20.0,
        "Informational": 1, "Informational_Duration": 10.0,
        "ProductRelated": 7, "ProductRelated_Duration": 70.0,
        "BounceRates": 0.0, "ExitRates": 0.1, "PageValues": 0.0, "VisitorType": "New_Visitor",
    }])
    out = add_features(df).iloc[0]
    assert out["TotalPages"] == 10 and out["TotalDuration"] == 100
    assert out["AvgTimePerPage"] == pytest.approx(10.0)
    assert out["ProductPageShare"] == pytest.approx(0.7)
    assert out["IsNewVisitor"] == 1


def test_zero_page_session_has_no_nan():
    df = pd.DataFrame([{
        "Administrative": 0, "Administrative_Duration": 0.0, "Informational": 0,
        "Informational_Duration": 0.0, "ProductRelated": 0, "ProductRelated_Duration": 0.0,
        "BounceRates": 0.2, "ExitRates": 0.2, "PageValues": 0.0, "VisitorType": "Other",
    }])
    X = get_cluster_matrix(df)
    assert list(X.columns) == CLUSTER_FEATURES
    assert not X.isna().any().any()

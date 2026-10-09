import numpy as np
import pandas as pd
import pytest

from src.clustering import (evaluate_k_range, load_artifacts, predict_segments, save_artifacts,
                            scale_features, suggest_k, train_kmeans)
from src.evaluation import cluster_quality, name_segments, profile_clusters
from src.feature_engineering import add_features, get_cluster_matrix


@pytest.fixture(scope="module")
def blobs():
    rng = np.random.default_rng(1)
    centers = np.array([[0, 0], [8, 8], [-8, 8]])
    X = np.vstack([c + rng.normal(size=(100, 2)) for c in centers])
    return X


def test_scale_features_zero_mean_unit_std(blobs):
    Xs, scaler = scale_features(pd.DataFrame(blobs))
    assert np.allclose(Xs.mean(axis=0), 0, atol=1e-8)
    assert np.allclose(Xs.std(axis=0), 1, atol=1e-8)


def test_evaluate_k_range_finds_true_k(blobs):
    Xs, _ = scale_features(pd.DataFrame(blobs))
    res = evaluate_k_range(Xs, range(2, 6))
    assert list(res["k"]) == [2, 3, 4, 5]
    assert res["inertia"].is_monotonic_decreasing
    assert suggest_k(res, min_k=2) == 3


def test_train_is_deterministic(blobs):
    Xs, _ = scale_features(pd.DataFrame(blobs))
    a, b = train_kmeans(Xs, 3), train_kmeans(Xs, 3)
    assert np.array_equal(a.labels_, b.labels_)


def test_save_load_roundtrip(tmp_path, blobs):
    Xs, scaler = scale_features(pd.DataFrame(blobs))
    km = train_kmeans(Xs, 3)
    save_artifacts(km, scaler, {0: {"name": "A"}}, tmp_path)
    km2, sc2, info = load_artifacts(tmp_path)
    assert np.array_equal(km.predict(Xs), km2.predict(Xs))
    assert info[0]["name"] == "A"


def _visitor(**kw):
    base = dict(Administrative=0, Administrative_Duration=0.0, Informational=0, Informational_Duration=0.0,
                ProductRelated=1, ProductRelated_Duration=0.0, BounceRates=0.2, ExitRates=0.2,
                PageValues=0.0, VisitorType="Returning_Visitor")
    base.update(kw)
    return pd.DataFrame([base])


@pytest.fixture(scope="module")
def real_artifacts():
    return load_artifacts("models")


def test_saved_model_segments_known_profiles(real_artifacts):
    model, scaler, info = real_artifacts
    name = lambda df: info[int(predict_segments(df, model, scaler)[0])]["name"]
    assert name(_visitor()) == "Bouncers"
    buyer = _visitor(Administrative=6, Administrative_Duration=200.0, Informational=3,
                     Informational_Duration=120.0, ProductRelated=60, ProductRelated_Duration=2400.0,
                     BounceRates=0.005, ExitRates=0.02, PageValues=28.0)
    assert name(buyer) == "High-Value Buyers"


def test_saved_model_matches_training_data(real_artifacts):
    model, scaler, info = real_artifacts
    df = pd.read_csv("data/processed/visitors_cleaned.csv")
    labels = predict_segments(df, model, scaler)
    assert np.array_equal(labels, model.labels_)
    assert set(info) == set(range(model.n_clusters))


def test_segment_names_are_unique_and_profile_is_consistent(real_artifacts):
    model, scaler, info = real_artifacts
    df = add_features(pd.read_csv("data/processed/visitors_cleaned.csv"))
    prof = profile_clusters(df, model.labels_)
    assert prof["visitors"].sum() == len(df)
    names = [v["name"] for v in name_segments(prof).values()]
    assert len(names) == len(set(names))
    q = cluster_quality(scaler.transform(get_cluster_matrix(df)), model.labels_)
    assert q["silhouette"] > 0.2

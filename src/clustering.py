"""K-Means clustering utilities: scaling, choosing k, training, persistence, prediction."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.feature_engineering import get_cluster_matrix

MODELS_DIR = Path("models")
RANDOM_STATE = 42


def scale_features(X: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler


def evaluate_k_range(
    X_scaled: np.ndarray,
    k_range: range = range(2, 11),
    random_state: int = RANDOM_STATE,
    silhouette_sample: int = 5000,
) -> pd.DataFrame:
    """Fit K-Means for every k and collect inertia + quality scores."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(X_scaled)
        rows.append(
            {
                "k": k,
                "inertia": km.inertia_,
                "silhouette": silhouette_score(
                    X_scaled, km.labels_, sample_size=min(silhouette_sample, len(X_scaled)),
                    random_state=random_state,
                ),
                "davies_bouldin": davies_bouldin_score(X_scaled, km.labels_),
            }
        )
    return pd.DataFrame(rows)


def suggest_k(results: pd.DataFrame, min_k: int = 3) -> int:
    """Pick the k with the best silhouette, ignoring k < min_k.

    k=2 usually wins on silhouette but is too coarse to be actionable for
    marketing, so we require at least `min_k` segments.
    """
    eligible = results[results["k"] >= min_k]
    return int(eligible.loc[eligible["silhouette"].idxmax(), "k"])


def train_kmeans(X_scaled: np.ndarray, k: int, random_state: int = RANDOM_STATE) -> KMeans:
    return KMeans(n_clusters=k, n_init=20, random_state=random_state).fit(X_scaled)


def save_artifacts(
    model: KMeans,
    scaler: StandardScaler,
    segment_info: dict | None = None,
    models_dir: str | Path = MODELS_DIR,
) -> None:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / "kmeans_model.joblib")
    joblib.dump(scaler, models_dir / "scaler.joblib")
    if segment_info is not None:
        with open(models_dir / "segment_info.json", "w") as f:
            json.dump({str(k): v for k, v in segment_info.items()}, f, indent=2)


def load_artifacts(models_dir: str | Path = MODELS_DIR):
    """Return (model, scaler, segment_info). segment_info is {} if not saved."""
    models_dir = Path(models_dir)
    model = joblib.load(models_dir / "kmeans_model.joblib")
    scaler = joblib.load(models_dir / "scaler.joblib")
    info_path = models_dir / "segment_info.json"
    info = {}
    if info_path.exists():
        with open(info_path) as f:
            info = {int(k): v for k, v in json.load(f).items()}
    return model, scaler, info


def predict_segments(df: pd.DataFrame, model: KMeans, scaler: StandardScaler) -> np.ndarray:
    """Assign raw/cleaned visitor rows to clusters with the trained artifacts."""
    X = get_cluster_matrix(df)
    return model.predict(scaler.transform(X))

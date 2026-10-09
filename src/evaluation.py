"""Cluster evaluation, profiling, segment naming and plotting helpers."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

FIGURES_DIR = Path("reports/figures")
sns.set_theme(style="whitegrid")


# --------------------------------------------------------------------------- #
# Metrics & profiling
# --------------------------------------------------------------------------- #
def cluster_quality(X_scaled: np.ndarray, labels: np.ndarray, sample: int = 5000) -> dict:
    """Internal validity metrics for a clustering."""
    return {
        "silhouette": float(
            silhouette_score(X_scaled, labels, sample_size=min(sample, len(X_scaled)), random_state=42)
        ),
        "davies_bouldin": float(davies_bouldin_score(X_scaled, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X_scaled, labels)),
    }


def profile_clusters(df_features: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Mean behaviour per cluster plus size and (post-hoc) purchase rate."""
    d = df_features.copy()
    d["cluster"] = labels
    prof = d.groupby("cluster").agg(
        visitors=("cluster", "size"),
        avg_pages=("TotalPages", "mean"),
        avg_duration_sec=("TotalDuration", "mean"),
        avg_time_per_page=("AvgTimePerPage", "mean"),
        product_page_share=("ProductPageShare", "mean"),
        bounce_rate=("BounceRates", "mean"),
        exit_rate=("ExitRates", "mean"),
        page_value=("PageValues", "mean"),
        new_visitor_share=("IsNewVisitor", "mean"),
        conversion_rate=("Revenue", "mean"),
    )
    prof.insert(1, "share", prof["visitors"] / prof["visitors"].sum())
    return prof


SEGMENT_PLAYBOOK = {
    "Bouncers": (
        "Land on the site and leave almost immediately: ~2 pages, a few seconds, very high bounce rate.",
        "Improve landing-page relevance and load speed; review the traffic sources sending them.",
    ),
    "New Visitor Prospects": (
        "First-time visitors with solid engagement and a surprisingly high purchase rate.",
        "Welcome offers, onboarding content and email capture to turn them into returning customers.",
    ),
    "High-Value Buyers": (
        "Returning visitors who view lots of pages, reach high-value pages and convert at the highest rate.",
        "Protect the experience: loyalty perks, cross-sell/upsell, and a frictionless checkout.",
    ),
    "Engaged Browsers": (
        "Returning visitors who browse deeply for a long time but rarely reach valuable pages.",
        "Retarget with reminders, comparison tools and limited-time incentives to close the gap.",
    ),
    "Light Browsers": (
        "Returning visitors with short, shallow sessions and low purchase intent.",
        "Surface bestsellers and personalised recommendations to deepen the visit.",
    ),
}


def name_segments(profile: pd.DataFrame) -> dict[int, dict]:
    """Give each cluster a business name using simple, deterministic rules.

    The rules rank clusters on their profile (not on cluster ids, which are arbitrary),
    so names stay correct when the model is retrained. Written for k = 5; clusters left
    unassigned (other k) fall back to "Segment <id>".
    """
    remaining = set(profile.index)
    assigned: dict[int, str] = {}

    def take(name: str, series: pd.Series) -> None:
        if not remaining:
            return
        pick = series.loc[list(remaining)].idxmax()
        assigned[int(pick)] = name
        remaining.discard(pick)

    take("Bouncers", profile["bounce_rate"])
    take("New Visitor Prospects", profile["new_visitor_share"])
    take("High-Value Buyers", profile["page_value"])
    take("Engaged Browsers", profile["avg_pages"])
    take("Light Browsers", -profile["avg_pages"])  # whatever is left

    info = {}
    for cid in profile.index:
        name = assigned.get(int(cid), f"Segment {cid}")
        desc, action = SEGMENT_PLAYBOOK.get(name, ("", ""))
        row = profile.loc[cid]
        info[int(cid)] = {
            "name": name,
            "description": desc,
            "action": action,
            "visitors": int(row["visitors"]),
            "share": round(float(row["share"]), 4),
            "conversion_rate": round(float(row["conversion_rate"]), 4),
        }
    return info


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _save(fig: plt.Figure, path: str | Path | None) -> None:
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")


def plot_distributions(df: pd.DataFrame, path: str | Path | None = FIGURES_DIR / "distributions.png"):
    cols = [
        "ProductRelated", "ProductRelated_Duration", "Administrative",
        "BounceRates", "ExitRates", "PageValues",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.ravel(), cols):
        sns.histplot(df[col], bins=40, ax=ax, color="#4C72B0")
        ax.set_yscale("log")
        ax.set_title(col)
        ax.set_ylabel("count (log)")
    fig.suptitle("Distributions of key behavioural variables (heavily right-skewed)", fontsize=14)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_correlation(df: pd.DataFrame, path: str | Path | None = FIGURES_DIR / "correlation_matrix.png"):
    corr = df.select_dtypes("number").corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f",
                annot_kws={"size": 7}, square=True, ax=ax)
    ax.set_title("Correlation matrix")
    _save(fig, path)
    return fig


def plot_elbow(results: pd.DataFrame, chosen_k: int | None = None,
               path: str | Path | None = FIGURES_DIR / "elbow_curve.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results["k"], results["inertia"], "o-", color="#4C72B0")
    if chosen_k is not None:
        ax.axvline(chosen_k, ls="--", color="crimson", label=f"chosen k = {chosen_k}")
        ax.legend()
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia (within-cluster SSE)")
    ax.set_title("Elbow method")
    _save(fig, path)
    return fig


def plot_silhouette(results: pd.DataFrame, chosen_k: int | None = None,
                    path: str | Path | None = FIGURES_DIR / "silhouette_score.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results["k"], results["silhouette"], "o-", color="#55A868")
    if chosen_k is not None:
        ax.axvline(chosen_k, ls="--", color="crimson", label=f"chosen k = {chosen_k}")
        ax.legend()
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Mean silhouette score")
    ax.set_title("Silhouette analysis")
    _save(fig, path)
    return fig


def plot_clusters(X_scaled: np.ndarray, labels: np.ndarray, names: dict[int, dict] | None = None,
                  path: str | Path | None = FIGURES_DIR / "clusters.png"):
    """2-D PCA projection of the clusters."""
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(9, 7))
    for cid in sorted(np.unique(labels)):
        m = labels == cid
        label = names[cid]["name"] if names else f"Cluster {cid}"
        ax.scatter(Z[m, 0], Z[m, 1], s=8, alpha=0.5, label=f"{label} (n={m.sum():,})")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} variance)")
    ax.set_title("Visitor segments (PCA projection)")
    ax.legend(markerscale=3)
    _save(fig, path)
    return fig


def plot_cluster_heatmap(profile: pd.DataFrame, names: dict[int, dict] | None = None, path=None):
    """Standardised profile heatmap: which features define each segment."""
    cols = ["avg_pages", "avg_duration_sec", "avg_time_per_page", "bounce_rate",
            "exit_rate", "page_value", "new_visitor_share", "conversion_rate"]
    z = (profile[cols] - profile[cols].mean()) / profile[cols].std(ddof=0)
    if names:
        z.index = [names[i]["name"] for i in z.index]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.heatmap(z, annot=profile[cols].values if names is None else None, cmap="RdBu_r",
                center=0, ax=ax, cbar_kws={"label": "z-score across segments"})
    ax.set_title("Segment profiles (relative to other segments)")
    _save(fig, path)
    return fig

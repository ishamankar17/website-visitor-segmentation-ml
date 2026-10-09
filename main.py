"""End-to-end pipeline: raw CSV -> cleaned data -> features -> K-Means -> saved artifacts + figures.

Usage:
    python main.py            # uses k = 5
    python main.py --k 4      # choose another number of clusters
"""
import argparse
import os

os.environ.setdefault("MPLBACKEND", "Agg")  # headless plotting

from src import clustering, data_preprocessing, evaluation, feature_engineering  # noqa: E402


def main(k: int = 5, k_min: int = 2, k_max: int = 10) -> None:
    print("1/5  Cleaning data ...")
    df = data_preprocessing.run()
    print(f"     {len(df):,} cleaned sessions -> {data_preprocessing.PROCESSED_PATH}")

    print("2/5  Engineering features ...")
    feats = feature_engineering.add_features(df)
    X = feature_engineering.get_cluster_matrix(feats)
    X_scaled, scaler = clustering.scale_features(X)

    print("3/5  Searching for the optimal k (elbow + silhouette) ...")
    results = clustering.evaluate_k_range(X_scaled, range(k_min, k_max + 1))
    print(results.round(3).to_string(index=False))
    print(f"     Best silhouette with k >= 4: k = {clustering.suggest_k(results, min_k=4)} | using k = {k}")
    evaluation.plot_elbow(results, k)
    evaluation.plot_silhouette(results, k)

    print("4/5  Training final K-Means ...")
    model = clustering.train_kmeans(X_scaled, k)
    labels = model.labels_
    quality = evaluation.cluster_quality(X_scaled, labels)
    print("     Quality:", {m: round(v, 3) for m, v in quality.items()})

    profile = evaluation.profile_clusters(feats, labels)
    segment_info = evaluation.name_segments(profile)
    evaluation.plot_distributions(df)
    evaluation.plot_correlation(feats[[c for c in feats.select_dtypes("number").columns if not c.startswith("log_")]])
    evaluation.plot_clusters(X_scaled, labels, segment_info)

    print("5/5  Saving artifacts ...")
    clustering.save_artifacts(model, scaler, segment_info)
    out = profile.copy()
    out.index = [segment_info[i]["name"] for i in out.index]
    print(out.round(3).T.to_string())
    print("Done. Models in models/, figures in reports/figures/.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5, help="number of clusters for the final model")
    args = parser.parse_args()
    main(k=args.k)

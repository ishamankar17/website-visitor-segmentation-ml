# Website Visitor Segmentation

End-to-end machine-learning project that groups website visitors into behavioural
segments with **K-Means**, using the UCI
[Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
dataset (12,330 sessions), and serves the model through a **Streamlit** app.

## Results

| Segment | Share of visitors | Purchase rate | Profile |
|---|---|---|---|
| High-Value Buyers | 17% | 54.6% | Returning, ~61 pages, high page values |
| New Visitor Prospects | 13% | 23.7% | First-time visitors, decent engagement |
| Engaged Browsers | 42% | 6.2% | Returning, deep browsing, little value reached |
| Light Browsers | 22% | 1.8% | Short, shallow sessions |
| Bouncers | 5% | 0.5% | ~2 pages, a few seconds, very high bounce rate |

Purchases (`Revenue`) are **not** used to build the clusters – they are only used afterwards to
validate them. Cluster quality: silhouette 0.32, Davies-Bouldin 1.05 (k = 5).
The 5 segments account for 60% / 20% / 17% / 3% / 0.2% of all purchases respectively.

> Numbers come from 12,205 sessions after removing 125 duplicates. Re-running `python main.py`
> reproduces them (fixed random seed).

## Pipeline

1. **Clean** – drop duplicates, fix `June`→`Jun`, validate ranges, 0/1 booleans.
2. **Explore** – distributions are heavily right-skewed; bounce/exit rates are almost collinear.
3. **Engineer features** – total pages/duration, time per page, product-page share,
   new-visitor flag, `log1p` transforms of skewed variables.
4. **Scale** – `StandardScaler`.
5. **Choose k** – Elbow + Silhouette over k = 2…10. Silhouette alone favours k = 2–3 (too coarse),
   k = 5 and 6 are nearly tied (0.322 vs 0.333); **k = 5** was chosen at the elbow as the smallest
   number of segments that still separates buyers from browsers.
6. **Train & name segments** – final `KMeans`, names assigned by rules on each cluster's profile.
7. **Serve** – Streamlit app (single visitor, batch CSV upload, segment overview).

## Project structure

```
data/raw/            original CSV            data/processed/   cleaned CSV
notebooks/           01 EDA → 05 cluster analysis (executed, with outputs)
src/                 data_preprocessing · feature_engineering · clustering · evaluation
models/              kmeans_model.joblib · scaler.joblib · segment_info.json
reports/figures/     distributions · correlation_matrix · elbow_curve · silhouette_score · clusters
app/streamlit_app.py Streamlit UI
tests/               pytest suite
main.py              runs the whole pipeline
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

python main.py                       # clean → features → k-search → train → save model + figures
pytest -q                            # run tests
streamlit run app/streamlit_app.py   # http://localhost:8501
```

`python main.py --k 4` trains with a different number of clusters. Segment names/actions are written
for k = 5; for other k, unmatched clusters are labelled "Segment N".

## Docker

```bash
docker build -t visitor-segmentation .
docker run -p 8501:8501 visitor-segmentation
```

## GitHub

```bash
git init
git add .
git commit -m "Website visitor segmentation project"
git branch -M main
git remote add origin https://github.com/<your-user>/website-visitor-segmentation.git
git push -u origin main
```

`.github/workflows/ci.yml` runs the tests and builds the Docker image on every push / PR.

## Deployment

**Streamlit Community Cloud (free, easiest)**
1. Push the repo to GitHub (the trained model and processed data are committed, so no training step is needed).
2. Go to <https://share.streamlit.io> → *New app* → pick the repo, branch `main`, main file `app/streamlit_app.py`.
3. Deploy. Streamlit installs `requirements.txt` automatically.

**Any container host (Render, Railway, Fly.io, Google Cloud Run, Azure, AWS App Runner)**
Point the service at this repo's `Dockerfile`; it listens on port 8501.

## Limitations

- The dataset is a single e-commerce site; segments may not transfer to other sites.
- Clusters overlap in feature space (see `reports/figures/clusters.png`) – behaviour is a continuum.
- K-Means assumes roughly spherical clusters on scaled features; Gaussian mixtures or
  HDBSCAN are natural next experiments.
- Segment names are heuristics layered on top of the clusters, not ground truth.

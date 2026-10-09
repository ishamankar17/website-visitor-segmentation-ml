"""Streamlit app: segment a website visitor from their session behaviour."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.clustering import load_artifacts, predict_segments  # noqa: E402
from src.feature_engineering import INPUT_COLS  # noqa: E402

st.set_page_config(page_title="Visitor Segmentation", page_icon="🛒", layout="wide")

PRESETS = {
    "Custom": None,
    "Quick bounce": dict(admin=0, admin_d=0, info=0, info_d=0, prod=1, prod_d=0,
                         bounce=0.2, exit=0.2, pv=0.0, vtype="Returning_Visitor"),
    "Casual browser": dict(admin=1, admin_d=20, info=0, info_d=0, prod=8, prod_d=200,
                           bounce=0.02, exit=0.06, pv=0.0, vtype="Returning_Visitor"),
    "Deep researcher": dict(admin=4, admin_d=120, info=2, info_d=80, prod=45, prod_d=1800,
                            bounce=0.01, exit=0.025, pv=0.0, vtype="Returning_Visitor"),
    "Ready-to-buy customer": dict(admin=6, admin_d=200, info=3, info_d=120, prod=60, prod_d=2400,
                                  bounce=0.005, exit=0.02, pv=28.0, vtype="Returning_Visitor"),
    "Promising newcomer": dict(admin=2, admin_d=60, info=1, info_d=30, prod=18, prod_d=700,
                               bounce=0.002, exit=0.018, pv=8.0, vtype="New_Visitor"),
}


@st.cache_resource
def get_artifacts():
    return load_artifacts(ROOT / "models")


@st.cache_data
def get_reference_data():
    path = ROOT / "data" / "processed" / "visitors_cleaned.csv"
    return pd.read_csv(path) if path.exists() else None


try:
    model, scaler, info = get_artifacts()
except FileNotFoundError:
    st.error("Model files not found. Run `python main.py` first to train and save the model.")
    st.stop()


def segment_name(cid: int) -> str:
    return info.get(cid, {}).get("name", f"Segment {cid}")


def preset_to_row(p: dict) -> dict:
    """Convert a PRESETS entry into the column names the model expects."""
    return {
        "Administrative": p["admin"], "Administrative_Duration": p["admin_d"],
        "Informational": p["info"], "Informational_Duration": p["info_d"],
        "ProductRelated": p["prod"], "ProductRelated_Duration": p["prod_d"],
        "BounceRates": p["bounce"], "ExitRates": p["exit"],
        "PageValues": p["pv"], "VisitorType": p["vtype"],
    }


@st.cache_data
def get_segmented_reference():
    ref = get_reference_data()
    if ref is None or any(c not in ref.columns for c in INPUT_COLS):
        return None
    ref = ref.copy()
    ref["SegmentID"] = predict_segments(ref, model, scaler)
    ref["Segment"] = ref["SegmentID"].map(segment_name)
    return ref


st.title("🛒 Website Visitor Segmentation")
st.caption("K-Means segments built on the UCI Online Shoppers Purchasing Intention dataset.")

tab_one, tab_batch, tab_overview, tab_compare, tab_whatif, tab_explore = st.tabs(
    ["Single visitor", "Batch upload", "Segments overview",
     "Compare visitors", "What-if explorer", "Data explorer"]
)

# --------------------------------------------------------------------------- #
with tab_one:
    preset_name = st.selectbox("Start from a preset", list(PRESETS))
    p = PRESETS[preset_name] or dict(admin=2, admin_d=60, info=0, info_d=0, prod=20, prod_d=600,
                                     bounce=0.02, exit=0.04, pv=0.0, vtype="Returning_Visitor")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Administrative pages")
        admin = st.number_input("Pages visited", 0, 100, p["admin"], key=f"a{preset_name}")
        admin_d = st.number_input("Seconds spent", 0.0, 10000.0, float(p["admin_d"]), key=f"ad{preset_name}")
        st.subheader("Informational pages")
        info_n = st.number_input("Pages visited ", 0, 100, p["info"], key=f"i{preset_name}")
        info_d = st.number_input("Seconds spent ", 0.0, 10000.0, float(p["info_d"]), key=f"id{preset_name}")
    with c2:
        st.subheader("Product pages")
        prod = st.number_input("Pages visited  ", 0, 1000, p["prod"], key=f"p{preset_name}")
        prod_d = st.number_input("Seconds spent  ", 0.0, 100000.0, float(p["prod_d"]), key=f"pd{preset_name}")
        vtype = st.selectbox(
            "Visitor type", ["Returning_Visitor", "New_Visitor", "Other"],
            index=["Returning_Visitor", "New_Visitor", "Other"].index(p["vtype"]), key=f"v{preset_name}",
        )
    with c3:
        st.subheader("Session quality")
        bounce = st.slider("Bounce rate", 0.0, 0.2, float(p["bounce"]), 0.001, key=f"b{preset_name}")
        exit_ = st.slider("Exit rate", 0.0, 0.2, float(p["exit"]), 0.001, key=f"e{preset_name}")
        pv = st.slider("Page value", 0.0, 400.0, float(p["pv"]), 0.5, key=f"pv{preset_name}")

    visitor = pd.DataFrame([{
        "Administrative": admin, "Administrative_Duration": admin_d,
        "Informational": info_n, "Informational_Duration": info_d,
        "ProductRelated": prod, "ProductRelated_Duration": prod_d,
        "BounceRates": bounce, "ExitRates": exit_, "PageValues": pv, "VisitorType": vtype,
    }])
    cid = int(predict_segments(visitor, model, scaler)[0])
    seg = info.get(cid, {})

    st.divider()
    st.success(f"### Segment: {segment_name(cid)}")
    if seg:
        m1, m2, m3 = st.columns(3)
        m1.metric("Share of all visitors", f"{seg['share']:.1%}")
        m2.metric("Typical purchase rate", f"{seg['conversion_rate']:.1%}")
        m3.metric("Segment size", f"{seg['visitors']:,}")
        st.markdown(f"**Who they are:** {seg['description']}")
        st.markdown(f"**Recommended action:** {seg['action']}")

# --------------------------------------------------------------------------- #
with tab_batch:
    st.write("Upload a CSV with these columns: " + ", ".join(f"`{c}`" for c in INPUT_COLS))
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        data = pd.read_csv(up)
        missing = [c for c in INPUT_COLS if c not in data.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            data["SegmentID"] = predict_segments(data, model, scaler)
            data["Segment"] = data["SegmentID"].map(segment_name)
            st.dataframe(data.head(50), width="stretch")
            counts = data["Segment"].value_counts()
            st.bar_chart(counts)
            st.download_button("Download segmented CSV", data.to_csv(index=False).encode(),
                               "visitors_segmented.csv", "text/csv")

# --------------------------------------------------------------------------- #
with tab_overview:
    rows = [
        {"Segment": v["name"], "Visitors": v["visitors"], "Share": f"{v['share']:.1%}",
         "Purchase rate": f"{v['conversion_rate']:.1%}", "Recommended action": v["action"]}
        for _, v in sorted(info.items())
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    fig = ROOT / "reports" / "figures" / "clusters.png"
    if fig.exists():
        st.image(str(fig), caption="PCA projection of the segments")

# --------------------------------------------------------------------------- #
# NEW: compare two visitors side by side
with tab_compare:
    st.write("Pick two preset visitors and see how they land in different segments.")
    names = [n for n in PRESETS if n != "Custom"]
    col_a, col_b = st.columns(2)
    sel_a = col_a.selectbox("Visitor A", names, index=0, key="cmp_a")
    sel_b = col_b.selectbox("Visitor B", names, index=len(names) - 1, key="cmp_b")

    cmp_rows = pd.DataFrame([preset_to_row(PRESETS[sel_a]), preset_to_row(PRESETS[sel_b])])
    cmp_ids = predict_segments(cmp_rows, model, scaler)

    for col, label, cid_ in zip((col_a, col_b), (sel_a, sel_b), cmp_ids):
        s = info.get(int(cid_), {})
        with col:
            st.success(f"**{label}** → {segment_name(int(cid_))}")
            if s:
                st.metric("Typical purchase rate", f"{s['conversion_rate']:.1%}")
                st.caption(s["action"])

    st.dataframe(cmp_rows.T.rename(columns={0: sel_a, 1: sel_b}), width="stretch")

# --------------------------------------------------------------------------- #
# NEW: what-if explorer, sweeping one feature of the Single visitor input
with tab_whatif:
    st.write("Take the visitor from the **Single visitor** tab and vary one "
             "feature to see where the segment flips.")
    SWEEP = {
        "ProductRelated": (0, 300), "ProductRelated_Duration": (0, 10000),
        "Administrative": (0, 30), "Informational": (0, 20),
        "BounceRates": (0.0, 0.2), "ExitRates": (0.0, 0.2), "PageValues": (0.0, 400.0),
    }
    feat = st.selectbox("Feature to vary", list(SWEEP))
    lo, hi = SWEEP[feat]
    lo, hi = st.slider("Range", float(lo), float(hi), (float(lo), float(hi)))
    steps = st.slider("Resolution (points)", 10, 200, 60)

    values = np.linspace(lo, hi, steps)
    sweep = pd.concat([visitor] * steps, ignore_index=True)
    sweep[feat] = values
    sweep_ids = predict_segments(sweep, model, scaler)
    sweep_names = [segment_name(int(i)) for i in sweep_ids]

    # Collapse consecutive identical segments into ranges
    ranges, start = [], 0
    for i in range(1, steps + 1):
        if i == steps or sweep_names[i] != sweep_names[start]:
            ranges.append({"From": round(values[start], 4), "To": round(values[i - 1], 4),
                           "Segment": sweep_names[start]})
            start = i
    st.dataframe(pd.DataFrame(ranges), width="stretch", hide_index=True)

    st.scatter_chart(pd.DataFrame({feat: values, "Segment": sweep_names}),
                     x=feat, y="Segment", color="Segment")
    st.caption(f"All other inputs are held at the values from the Single visitor tab "
               f"(currently: {segment_name(cid)}).")

# --------------------------------------------------------------------------- #
# NEW: data explorer over the reference dataset
with tab_explore:
    ref = get_segmented_reference()
    if ref is None:
        st.info("Reference data not found (data/processed/visitors_cleaned.csv) "
                "or it is missing the model input columns.")
    else:
        seg_options = sorted(ref["Segment"].unique())
        chosen = st.multiselect("Segments to show", seg_options, default=seg_options)
        view = ref[ref["Segment"].isin(chosen)]

        if "VisitorType" in view.columns:
            vt_options = sorted(view["VisitorType"].unique())
            vt = st.multiselect("Visitor type", vt_options, default=vt_options)
            view = view[view["VisitorType"].isin(vt)]

        st.metric("Visitors in selection", f"{len(view):,}")

        if view.empty:
            st.warning("No visitors match the current filters.")
        else:
            num_cols = [c for c in INPUT_COLS if pd.api.types.is_numeric_dtype(view[c])]
            x_col, y_col = st.columns(2)
            x = x_col.selectbox("X axis", num_cols,
                                index=num_cols.index("ProductRelated") if "ProductRelated" in num_cols else 0)
            y = y_col.selectbox("Y axis", num_cols,
                                index=num_cols.index("PageValues") if "PageValues" in num_cols else 0)

            sample = view.sample(min(len(view), 3000), random_state=0)
            st.scatter_chart(sample, x=x, y=y, color="Segment")

            st.subheader("Average behaviour per segment")
            st.dataframe(view.groupby("Segment")[num_cols].mean().round(2), width="stretch")

            st.download_button("Download filtered data", view.to_csv(index=False).encode(),
                               "filtered_visitors.csv", "text/csv")
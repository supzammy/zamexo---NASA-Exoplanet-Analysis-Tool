# ZAMEXO — Streamlit app
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from lightkurve import search_lightcurve

from utils.features import bls_features, fold_curve
from utils.nasa import resolve_features_for_target

warnings.filterwarnings("ignore")

st.set_page_config(page_title="ZAMEXO — Exoplanet Detector (Real NASA Data)", layout="wide")
st.title("ZAMEXO — Exoplanet Detector (Real NASA Data: KOI/TOI/K2)")


@st.cache_resource
def load_model_artifacts():
    mdir = Path("models")
    if (mdir / "rf_multi.joblib").exists():
        model_path = mdir / "rf_multi.joblib"
        cols_path = mdir / "multi_feature_cols.json"
    elif (mdir / "rf_koi.joblib").exists():
        model_path = mdir / "rf_koi.joblib"
        cols_path = mdir / "koi_feature_cols.json"
    else:
        return None, None, None
    pipe = joblib.load(model_path)
    cols = json.loads(Path(cols_path).read_text())
    inv_label = {0: "FALSE POSITIVE", 1: "CANDIDATE", 2: "CONFIRMED"}
    return pipe, cols, inv_label


def predict_with_model(vec: dict):
    pipe, cols, inv_label = load_model_artifacts()
    if pipe is None:
        return None
    alias = {
        "koi_period": "period",
        "koi_duration": "duration",
        "koi_depth": "depth",
        "koi_prad": "prad",
        "koi_steff": "teff",
        "koi_slogg": "logg",
        "koi_srad": "srad",
    }
    row = {}
    for c in cols:
        key = alias.get(c, c)
        row[c] = vec.get(key, np.nan)
    X = pd.DataFrame([row], columns=cols)
    proba = pipe.predict_proba(X)[0]
    i = int(np.argmax(proba))
    return {
        "label": inv_label.get(i, str(i)),
        "proba": {inv_label.get(j, str(j)): float(p) for j, p in enumerate(proba)},
    }


def render_prob_bar(prob_map: dict):
    labels = ["FALSE POSITIVE", "CANDIDATE", "CONFIRMED"]
    df = pd.DataFrame(
        {"class": labels, "probability": [prob_map.get(k, 0.0) for k in labels]}
    ).set_index("class")
    st.bar_chart(df)


def fetch_lightcurve(target: str, mission: str | None):
    sr = search_lightcurve(target, mission=mission if mission != "Auto" else None)
    if len(sr) == 0:
        return None
    lc = sr[0].download()
    if lc is None:
        return None
    time = np.array(lc.time.value, dtype=float)
    flux = np.array(lc.flux.value, dtype=float)
    return time, flux


def _bin_fold(phase: np.ndarray, flux: np.ndarray, nbins: int = 100):
    bins = np.linspace(-0.5, 0.5, nbins + 1)
    idx = np.digitize(phase, bins) - 1
    centers = 0.5 * (bins[1:] + bins[:-1])
    bmed = np.array(
        [np.nanmedian(flux[idx == k]) if np.any(idx == k) else np.nan for k in range(nbins)]
    )
    return centers, bmed


with st.sidebar:
    st.header("Target / Data")
    target = st.text_input("Target (e.g., Kepler-10, TOI 123)", value="")
    mission = st.selectbox("Mission", options=["Auto", "Kepler", "K2", "TESS"], index=0)
    if st.button("Clear data cache"):
        try:
            for p in Path("data").glob("*.csv"):
                p.unlink(missing_ok=True)
            st.toast("Cleared data/*.csv cache.", icon="🧹")
        except Exception as e:
            st.toast(f"Failed to clear: {e}", icon="⚠️")
    st.divider()
    st.header("BLS Settings")
    max_period = st.slider("Max period (days)", 2.0, 50.0, 30.0, 0.5)
    min_period = st.slider("Min period (days)", 0.1, 2.0, 0.2, 0.05)
    st.divider()
    st.header("Upload CSV")
    up = st.file_uploader("CSV with columns: time, flux", type=["csv"])
    st.caption("UI polish later; core logic is functional.")

tab1, tab2, tab3 = st.tabs(["Search target", "Upload CSV", "Explainability"])

with tab1:
    if target:
        with st.spinner("Fetching light curve…"):
            tf = fetch_lightcurve(target, mission)
        if tf is None:
            st.info("No light curve found.")
        else:
            time, flux = tf
            st.write(f"Points: {len(time)} | Span: {time.max() - time.min():.1f} d")
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(time, flux, "k.", ms=1)
            ax.set_xlabel("Time [d]")
            ax.set_ylabel("Flux")
            st.pyplot(fig, use_container_width=True)

            with st.spinner("Running BLS…"):
                feats = bls_features(time, flux, max_period=max_period, min_period=min_period)
            if np.isfinite(feats.get("period", np.nan)):
                st.subheader("BLS results")
                c1, c2, c3 = st.columns(3)
                c1.metric("Period (d)", f"{feats['period']:.4f}")
                c2.metric("Duration (d)", f"{feats['duration']:.4f}")
                c3.metric("Depth", f"{feats['depth']:.5f}")
                c1.metric("SDE", f"{feats['sde']:.2f}")
                c2.metric("Points", f"{feats['n_points']}")
                c3.metric("Span (d)", f"{feats['time_span']:.1f}")
                # Phase-folded
                ph, order = fold_curve(time, feats["period"], feats["t0"])
                fig2, ax2 = plt.subplots(figsize=(8, 3))
                ax2.plot(ph, flux[order], "k.", ms=1, alpha=0.4)
                bc, bf = _bin_fold(ph, flux[order], nbins=80)
                ax2.plot(bc, bf, "C1-", lw=2)
                ax2.set_xlabel("Phase")
                ax2.set_ylabel("Flux")
                st.pyplot(fig2, use_container_width=True)

            pred = None
            with st.spinner("Predicting…"):
                resolved = resolve_features_for_target(target)
                if resolved is not None:
                    f = resolved["features"]
                    pred = predict_with_model(
                        {
                            "period": f[0],
                            "duration": f[1],
                            "depth": f[2],
                            "prad": f[3],
                            "teff": f[4],
                            "logg": f[5],
                            "srad": f[6],
                        }
                    )
                else:
                    pred = predict_with_model(
                        {
                            "period": feats.get("period"),
                            "duration": feats.get("duration"),
                            "depth": feats.get("depth"),
                        }
                    )
            if pred:
                st.subheader(f"Prediction: {pred['label']}")
                render_prob_bar(pred["proba"])
                st.toast("Prediction ready", icon="✅")

with tab2:
    if up is not None:
        dfu = pd.read_csv(up)
        # Normalize header names
        lower_cols = [c.lower() for c in dfu.columns]
        dfu.columns = lower_cols
        if {"time", "flux"} <= set(dfu.columns):
            time = dfu["time"].to_numpy(dtype=float)
            flux = dfu["flux"].to_numpy(dtype=float)
            st.write(f"Uploaded: {len(time)} points")
            feats = bls_features(time, flux, max_period=max_period, min_period=min_period)
            st.write("BLS:", feats)
            ph, order = fold_curve(time, feats["period"], feats["t0"])
            fig3, ax3 = plt.subplots(figsize=(8, 3))
            ax3.plot(ph, flux[order], "k.", ms=1, alpha=0.4)
            bc, bf = _bin_fold(ph, flux[order], nbins=80)
            ax3.plot(bc, bf, "C1-", lw=2)
            ax3.set_xlabel("Phase")
            ax3.set_ylabel("Flux")
            st.pyplot(fig3, use_container_width=True)
            pred = predict_with_model(
                {"period": feats["period"], "duration": feats["duration"], "depth": feats["depth"]}
            )
            if pred:
                st.subheader(f"Prediction: {pred['label']}")
                render_prob_bar(pred["proba"])
        else:
            st.error("CSV must have 'time' and 'flux' columns.")

with tab3:
    pipe, cols, inv = load_model_artifacts()
    if pipe is None:
        st.info("Train a model first (make train-multi or make train-baseline).")
    else:
        if hasattr(pipe.named_steps.get("rf", None), "feature_importances_"):
            rf = pipe.named_steps["rf"]
            importances = pd.Series(rf.feature_importances_, index=cols).sort_values(
                ascending=False
            )
            st.subheader("Feature importances (RandomForest)")
            st.bar_chart(importances)
        if st.checkbox("Compute SHAP for a dummy instance (slow)"):
            try:
                import shap

                rf = pipe.named_steps["rf"]
                imp = pipe.named_steps["imputer"]
                x0 = pd.DataFrame([np.nan] * len(cols), index=cols).T
                x0_imp = imp.transform(x0)
                expl = shap.TreeExplainer(rf)
                sv = expl.shap_values(x0_imp)
                vals = np.abs(sv[1][0])
                shp = pd.Series(vals, index=cols).sort_values(ascending=False)
                st.subheader("SHAP (abs, class=CANDIDATE)")
                st.bar_chart(shp)
            except Exception as e:
                st.warning(f"SHAP not available or failed: {e}")

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
try:
    import shap  # type: ignore
    _HAS_SHAP = True
except Exception:  # ModuleNotFoundError or other
    shap = None    # type: ignore
    _HAS_SHAP = False

# Optional imports from your codebase (fall back if missing)
try:
    from utils.features import map_bls_to_feature_row  # uses FEATURE_COLS in tests
except Exception:

    def map_bls_to_feature_row(feature_cols, bls, bls_map=None, fill_stats=None):
        row = {}
        default_map = {
            "koi_period": "period",
            "koi_duration": "duration",
            "koi_depth": "depth",
            "koi_depth_frac": "depth",
        }
        m = {**default_map, **(bls_map or {})}
        for c in feature_cols:
            key = m.get(c, c)
            v = bls.get(key, np.nan)
            if c == "koi_depth_frac" and np.isfinite(v):
                v = abs(float(v))
            row[c] = float(v) if isinstance(v, (int, float, np.floating)) else np.nan
        return row


try:
    from utils.plots import (
        plot_importances as _plot_importances,
    )
    from utils.plots import (
        plot_lightcurve_with_transits as _plot_lc_tx,
    )
    from utils.plots import (
        plot_phase_folded as _plot_phase,
    )
except Exception:
    _plot_lc_tx = _plot_phase = _plot_importances = None


def plot_lightcurve(time: np.ndarray, flux: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(time, flux, lw=0.8)
    ax.set_xlabel("Time [d]")
    ax.set_ylabel("Flux [norm]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_lightcurve_with_transits(time, flux, bls) -> plt.Figure:
    if _plot_lc_tx:
        return _plot_lc_tx(time, flux, bls)
    # Fallback: mark phase locations if period is finite
    fig = plot_lightcurve(time, flux)
    if bls and np.isfinite(bls.get("period", np.nan)):
        ax = fig.axes[0]
        p = bls["period"]
        t0 = bls.get("t0", time[0])
        for k in range(int((time[-1] - time[0]) // p) + 1):
            x = t0 + k * p
            ax.axvline(x, color="crimson", alpha=0.2, lw=1)
    return fig


def plot_phase_folded(time, flux, bls, nbins=100) -> plt.Figure:
    if _plot_phase:
        return _plot_phase(time, flux, bls, nbins=nbins)
    # Fallback: simple phase-fold
    period = bls.get("period", np.nan)
    if not np.isfinite(period) or period <= 0:
        return plot_lightcurve(time, flux)
    phase = ((time - time.min()) % period) / period
    order = np.argsort(phase)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(phase[order], flux[order], ".", ms=2, alpha=0.6)
    ax.set_xlabel("Phase")
    ax.set_ylabel("Flux [norm]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_importances(importances, feature_cols):
    if _plot_importances:
        return _plot_importances(importances)
    if not importances:
        st.info("No feature importances available.")
        return
    s = pd.Series(importances, index=feature_cols).sort_values(ascending=True)
    st.bar_chart(s)

#streamlit run app/streamlit_app.
@st.cache_resource(show_spinner=False)
def _get_shap_explainer(_model):
    if not _HAS_SHAP or model is None:
        return None
    try:
        return shap.TreeExplainer(_model)
    except Exception:
        return None


def get_shap_plot(model, X):
    """Generate a SHAP force plot for a single-row DataFrame X.
    Returns (fig, shap_values, predicted_class_index) or (None, None, None) on failure.
    """
    if not _HAS_SHAP or shap is None:
        return None, None, None
    if model is None or X is None or X.empty or not hasattr(model, "predict_proba"):
        return None, None, None
    explainer = _get_shap_explainer(model)
    if explainer is None:
        return None, None, None
    try:
        shap_values = explainer.shap_values(X)
        proba = model.predict_proba(X)[0]
        predicted_class_index = int(np.argmax(proba))
        fig, ax = plt.subplots(figsize=(6, 2.2))
        shap.force_plot(
            explainer.expected_value[predicted_class_index],
            shap_values[predicted_class_index],
            X.iloc[0],
            matplotlib=True,
            show=False,
            ax=ax,
        )
        fig.tight_layout()
        return fig, shap_values, predicted_class_index
    except Exception:
        return None, None, None


@st.cache_data(show_spinner=False)
def load_artifacts():
    """
    Try to load model artifacts if present in ./models.
    Expects model.joblib and optional JSON sidecars feature_cols.json, feature_stats.json, importances.json.
    """
    model = None
    feature_cols = []
    feature_stats = {}
    importances = {}

    models_dir = Path("models")
    try:
        import joblib  # optional

        model_path = models_dir / "model.joblib"
        if model_path.exists():
            model = joblib.load(model_path)
    except Exception:
        model = None

    def load_json(p: Path):
        try:
            if p.exists():
                return json.loads(p.read_text())
        except Exception:
            return {}
        return {}

    feature_cols = load_json(models_dir / "feature_cols.json") or []
    feature_stats = load_json(models_dir / "feature_stats.json") or {}
    importances = load_json(models_dir / "importances.json") or {}

    return model, feature_cols, feature_stats, importances


def run_inference(model, feature_cols, feature_stats, bls):
    if model is None or not feature_cols:
        return None, {}, None
    fill = (feature_stats.get("mean") or {}) if isinstance(feature_stats, dict) else {}
    row = map_bls_to_feature_row(feature_cols, bls, fill_stats=fill)
    X = pd.DataFrame([row], columns=feature_cols).fillna(0)  # Fill NaNs for SHAP
    try:
        proba = model.predict_proba(X)[0]
        classes = list(
            getattr(model, "classes_", [f"class_{i}" for i in range(len(proba))])
        )
        scores = dict(zip(classes, map(float, proba)))
        label = classes[int(np.argmax(proba))]
        return label, scores, X
    except Exception:
        try:
            y = model.predict(X)[0]
            return str(y), {}, X
        except Exception:
            return None, {}, None


def simple_bls(time: np.ndarray, flux: np.ndarray, max_period: float) -> dict:
    """
    Minimal, fast placeholder if your real BLS is unavailable.
    Returns NaNs if not enough data.
    """
    if time is None or flux is None:
        return {
            "period": np.nan,
            "duration": np.nan,
            "depth": np.nan,
            "sde": np.nan,
            "t0": np.nan,
        }
    # Coerce to plain float arrays (handle astropy masked/time objects)
    t = np.asarray(getattr(time, 'value', time), dtype=float)
    f = np.asarray(getattr(flux, 'value', flux), dtype=float)
    if np.ma.isMaskedArray(f):
        f = np.asarray(f.filled(np.nan), dtype=float)
    if np.ma.isMaskedArray(t):
        t = np.asarray(t.filled(np.nan), dtype=float)
    m = np.isfinite(t) & np.isfinite(f)
    t, f = t[m], f[m]
    if t.size < 10:
        return {
            "period": np.nan,
            "duration": np.nan,
            "depth": np.nan,
            "sde": np.nan,
            "t0": np.nan,
        }
    # Normalize time baseline and flux median
    t = t - t.min()
    med = np.nanmedian(f)
    if np.isfinite(med) and med != 0:
        f = f / med - 1.0
    else:
        f = f - np.nanmean(f)
    # Use a naive periodicity guess via autocorrelation as a fallback
    n = min(f.size, 5000)
    f_work = f[:n] - np.nanmean(f[:n])
    try:
        ac = np.correlate(f_work, f_work, mode="full")[n - 1 :]
    except TypeError:
        # Fallback: if still masked or unsupported, bail gracefully
        return {
            "period": np.nan,
            "duration": np.nan,
            "depth": np.nan,
            "sde": np.nan,
            "t0": float(time.min()) if hasattr(time, '__len__') and len(time) else np.nan,
        }
    ac[0] = 0.0
    idx = int(np.argmax(ac[1:]) + 1)
    if idx <= 0 or idx >= len(t):
        return {
            "period": np.nan,
            "duration": np.nan,
            "depth": np.nan,
            "sde": np.nan,
            "t0": np.nan,
        }
    # Convert lag index to time difference
    # Use median cadence to map samples->days robustly
    dt = float(np.median(np.diff(t)))
    period = float(idx * dt)
    if not np.isfinite(period) or period <= 0 or period > max_period:
        period = np.nan
    # Rough depth estimate
    depth = float(np.nanpercentile(f, 2)) if np.isfinite(period) else np.nan
    return {
        "period": period,
        "duration": max(0.05, period * 0.05) if np.isfinite(period) else np.nan,
        "depth": depth,
        "sde": np.nan,
        "t0": float(time.min()),
    }


def summarize_lightcurve(time, flux):
    import numpy as np
    t = np.asarray(time)
    f = np.asarray(flux)
    m = np.isfinite(t) & np.isfinite(f)
    t, f = t[m], f[m]
    if t.size == 0:
        return {}
    med = np.median(f)
    mean = np.mean(f)
    std = np.std(f)
    mad = np.median(np.abs(f - med)) * 1.4826  # robust sigma
    dyn_range = (np.max(f) - np.min(f)) / med if med != 0 else np.nan
    dt = np.diff(np.sort(t))
    cadence = np.median(dt) if dt.size else np.nan
    baseline = t.max() - t.min() if t.size else 0
    # simple gap analysis
    largest_gap = dt.max() if dt.size else 0
    # outliers
    outlier_mask = np.abs(f - med) > 5 * mad if np.isfinite(mad) and mad > 0 else np.zeros_like(f, bool)
    outliers = int(outlier_mask.sum())
    frac_out = outliers / f.size if f.size else 0
    return {
        "baseline_days": baseline,
        "cadence_days": cadence,
        "n_points": int(f.size),
        "median_flux": float(med),
        "mean_flux": float(mean),
        "std_flux": float(std),
        "robust_rms": float(mad),
        "frac_rms": float(std / med) if med != 0 else np.nan,
        "dynamic_range": float(dyn_range),
        "largest_gap_days": float(largest_gap),
        "outliers": outliers,
        "outlier_fraction": frac_out,
    }


def quality_rating(stats):
    score = 0
    if stats["frac_rms"] < 0.02: score += 1
    if stats["outlier_fraction"] < 0.01: score += 1
    if stats["largest_gap_days"] < 0.5 * stats["baseline_days"]: score += 1
    return ["LOW", "MODERATE", "HIGH"][min(score, 2)]

def main():
    st.set_page_config(page_title="NASA Transit Search", layout="wide")
    st.title("🚀 NASA Transit Search & Explainable AI")

    # Inject light CSS for tighter layout
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.2rem;}
        .stMetric label {font-weight:600;}
        .small-note {font-size:0.75rem; opacity:0.7;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar controls
    with st.sidebar:
        st.subheader("Input")
        mission = st.selectbox("Mission", ["Auto", "Kepler", "TESS"], index=0)
        target = st.text_input("Target (e.g., Kepler-10)", "")
        max_period = st.slider("Max period [days]", 1.0, 50.0, 10.0, step=0.5)
        nbins = st.slider("Phase bins", 25, 300, 100, step=5)
        uploaded = st.file_uploader("Upload CSV (time,flux)", type=["csv"])
        run_btn = st.button("Run Analysis")
        enable_shap = st.checkbox(
            "Enable SHAP (slower)", value=False, help="Compute per-prediction explanation."
        )
        if not _HAS_SHAP and enable_shap:
            st.warning("SHAP library not installed in this environment.")
        st.caption("Model + transit feature pipeline demo. Upload or specify target.")
        if st.button("Clear Lightkurve Cache"):
            import shutil, os, pathlib
            cache_dir = pathlib.Path.home()/".lightkurve"/"cache"/"mastDownload"
            shutil.rmtree(cache_dir, ignore_errors=True)
            st.success("Cache cleared.")

    model, feature_cols, feature_stats, importances = load_artifacts()

    # --- Data ingestion priority: uploaded > NASA fetch > synthetic demo ---
    time_arr = flux_arr = None
    source_label = "(none)"

    # 1. Uploaded CSV
    if uploaded is not None:
        try:
            df_u = pd.read_csv(uploaded)
            if {"time", "flux"}.issubset(df_u.columns):
                time_arr = df_u["time"].to_numpy(float)
                flux_arr = df_u["flux"].to_numpy(float)
                source_label = f"Upload: {uploaded.name}"
            else:
                st.error("CSV must contain 'time' and 'flux' columns.")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

    # 2. NASA Lightkurve fetch (only if not already satisfied by upload)
    if time_arr is None and run_btn and target.strip():
        from lightkurve import search_lightcurve
        target_clean = target.strip()
        missions_to_try = [mission] if mission != "Auto" else ["Kepler", "K2", "TESS"]
        fetched = False
        for m in missions_to_try:
            try:
                sr = search_lightcurve(target_clean, mission=m)
                if len(sr) == 0:
                    continue
                # Download a limited number of products to keep latency reasonable
                pieces = []
                for prod in sr[:3]:
                    try:
                        lc_part = prod.download()
                        if lc_part is not None:
                            pieces.append(lc_part)
                    except Exception:
                        continue
                if not pieces:
                    continue
                try:
                    lc_full = pieces[0]
                    if len(pieces) > 1:
                        lc_full = lc_full.append(pieces[1:])  # lightkurve collection append
                    # Normalize if possible
                    flux_vals = lc_full.flux.value if hasattr(lc_full.flux, 'value') else np.asarray(lc_full.flux)
                    time_vals = lc_full.time.value if hasattr(lc_full.time, 'value') else np.asarray(lc_full.time)
                    # Basic clean
                    mask = np.isfinite(time_vals) & np.isfinite(flux_vals)
                    time_arr = time_vals[mask]
                    flux_arr = flux_vals[mask]
                    source_label = f"NASA: {target_clean} ({m})"
                    st.success(f"Fetched {len(time_arr)} points from {m} for {target_clean}.")
                    fetched = True
                    break
                except Exception as e:
                    st.warning(f"Failed assembling light curve for mission {m}: {e}")
            except Exception:
                continue
        if not fetched and time_arr is None:
            st.error("No matching NASA light curve found (Kepler/K2/TESS). Using synthetic demo.")

    # 3. Synthetic fallback
    if time_arr is None:
        rng = np.random.default_rng(42)
        time_arr = np.linspace(0, 12, 3000)
        flux_arr = 1 + 0.001 * np.sin(2 * np.pi * time_arr / 2.1) + 5e-4 * rng.standard_normal(time_arr.size)
        source_label = "Synthetic demo"

    bls = simple_bls(time_arr, flux_arr, max_period=max_period)
    depth_ppm = bls["depth"] * 1e6 if np.isfinite(bls["depth"]) else np.nan

    tabs = st.tabs([
        "Overview",
        "Detection",
        "AI Classification",
        "Explainability",
        "Upload",
    ])

    # --- Overview Tab ---
    with tabs[0]:
        st.subheader("Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Period [d]", f"{bls['period']:.4g}" if np.isfinite(bls["period"]) else "–")
        c2.metric("Duration [d]", f"{bls['duration']:.4g}" if np.isfinite(bls["duration"]) else "–")
        c3.metric(
            "Depth (frac)",
            f"{bls['depth']:.3g}" if np.isfinite(bls["depth"]) else "–",
            help=f"≈{depth_ppm:.0f} ppm" if np.isfinite(depth_ppm) else None,
        )
        c4.metric("SDE", f"{bls['sde']:.3g}" if np.isfinite(bls["sde"]) else "–")
        st.caption(f"Source: {source_label}")
        st.write("Light Curve")
        fig = plot_lightcurve_with_transits(time_arr, flux_arr, bls)
        st.pyplot(fig, clear_figure=True)

    # --- Detection Tab ---
    with tabs[1]:
        st.subheader("Transit Detection & Phase Fold")
        if np.isfinite(bls["period"]):
            fig_phase = plot_phase_folded(time_arr, flux_arr, bls, nbins=nbins)
            st.pyplot(fig_phase, clear_figure=True)
        else:
            st.info("No reliable period estimate; adjust max period or provide longer baseline.")

    # --- AI Classification Tab ---
    with tabs[2]:
        st.subheader("Model Prediction")
        label, probs, X_pred = run_inference(model, feature_cols, feature_stats, bls)
        if model is None:
            st.warning("Model artifacts not found. Train first (scripts/train_baseline.py).")
        elif label is None:
            st.info("Insufficient features for prediction.")
        else:
            max_class = max(probs, key=probs.get) if probs else label
            confidence = probs.get(max_class, 0.0) if probs else 0.0
            st.markdown(f"**Prediction:** `{label}`  |  Confidence: {confidence:.2%}")
            if probs:
                st.bar_chart(pd.Series(probs))
        if X_pred is not None:
            st.caption("Feature vector prepared for model inference.")

    # --- Explainability Tab ---
    with tabs[3]:
        st.subheader("Explainability")
        if not feature_cols:
            st.info("No feature columns available.")
        else:
            st.write("### Global Feature Importances")
            plot_importances(importances, feature_cols)
        if enable_shap:
            st.write("### Per‑Prediction SHAP")
            if not _HAS_SHAP:
                st.error("SHAP unavailable in environment.")
            else:
                _, probs_tmp, X_pred_tmp = run_inference(
                    model, feature_cols, feature_stats, bls
                )
                if X_pred_tmp is None:
                    st.info("No prediction context for SHAP.")
                else:
                    with st.spinner("Computing SHAP…"):
                        shap_fig, shap_values, cls_idx = get_shap_plot(model, X_pred_tmp)
                    if shap_fig:
                        st.pyplot(shap_fig, clear_figure=True)
                        # Top contributors summary
                        try:
                            sv = shap_values[cls_idx][0]
                            top_i = np.argsort(np.abs(sv))[-5:][::-1]
                            st.write(
                                "**Top contributing features:**",
                                [
                                    f"{feature_cols[i]} ({sv[i]:+.3g})"
                                    for i in top_i
                                ],
                            )
                        except Exception:
                            pass
                    else:
                        st.warning("Could not generate SHAP explanation.")
        else:
            st.info("Enable SHAP in the sidebar to compute per‑prediction explanations.")

    # --- Upload Tab ---
    with tabs[4]:
        st.subheader("Upload Workflow")
        st.write(
            "Upload a CSV with columns 'time' and 'flux' via the sidebar. It will automatically populate the pipeline (Overview → Detection → AI → Explainability)."
        )
        st.markdown(
            "<span class='small-note'>If you already uploaded a file, it is in use above.</span>",
            unsafe_allow_html=True,
        )
    stats = summarize_lightcurve(time_arr, flux_arr)
    with st.expander("Light Curve Stats", expanded=False):
        cols = st.columns(3)
        cols[0].metric("Baseline (d)", f"{stats['baseline_days']:.2f}")
        cols[1].metric("Cadence (s)", f"{stats['cadence_days']*86400:.0f}")
        cols[2].metric("Points", f"{stats['n_points']}")
        cols[0].metric("RMS (frac)", f"{stats['frac_rms']:.3%}")
        cols[1].metric("Robust RMS", f"{stats['robust_rms']:.2f}")
        cols[2].metric("Outliers", f"{stats['outliers']} ({stats['outlier_fraction']:.2%})")
    qr = quality_rating(stats)
    st.caption(f"Data quality: **{qr}**")


if __name__ == "__main__":
    # Simplified entrypoint for Streamlit execution: avoid demo auto-downloads which
    # were triggering ValueError: I/O operation on closed file under some reload cycles.
    try:
        main()
    except Exception as e:  # final guard so Streamlit can render the traceback nicely
        import logging
        logging.exception("Fatal error in main(): %s", e)

test-batch:
	@set -e; \\
	for f in tests/test_*.py; do \\
	  echo "Running $$f"; \\
	  pytest -q $$f || exit 1; \\
	done

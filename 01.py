import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

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


def plot_importances(importances):
    if _plot_importances:
        return _plot_importances(importances)
    if not importances:
        st.info("No feature importances available.")
        return
    s = pd.Series(importances).sort_values(ascending=True)
    st.bar_chart(s)


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
        return None, {}
    fill = (feature_stats.get("mean") or {}) if isinstance(feature_stats, dict) else {}
    row = map_bls_to_feature_row(feature_cols, bls, fill_stats=fill)
    X = pd.DataFrame([row], columns=feature_cols)
    try:
        proba = model.predict_proba(X)[0]
        classes = list(
            getattr(model, "classes_", [f"class_{i}" for i in range(len(proba))])
        )
        scores = dict(zip(classes, map(float, proba)))
        label = classes[int(np.argmax(proba))]
        return label, scores
    except Exception:
        try:
            y = model.predict(X)[0]
            return str(y), {}
        except Exception:
            return None, {}


def simple_bls(time: np.ndarray, flux: np.ndarray, max_period: float) -> dict:
    """
    Minimal, fast placeholder if your real BLS is unavailable.
    Returns NaNs if not enough data.
    """
    if time is None or flux is None or len(time) < 10:
        return {
            "period": np.nan,
            "duration": np.nan,
            "depth": np.nan,
            "sde": np.nan,
            "t0": np.nan,
        }
    # Use a naive periodicity guess via autocorrelation as a fallback
    t = time - time.min()
    f = flux - np.nanmedian(flux)
    n = min(len(f), 5000)
    f = f[:n] - np.mean(f[:n])
    ac = np.correlate(f, f, mode="full")[n - 1 :]
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
    depth = float(np.percentile(f, 2)) if np.isfinite(period) else np.nan
    return {
        "period": period,
        "duration": max(0.05, period * 0.05) if np.isfinite(period) else np.nan,
        "depth": depth,
        "sde": np.nan,
        "t0": float(time.min()),
    }


def main():
    st.set_page_config(page_title="NASA Project", layout="wide")
    st.title("NASA Transit Search")

    # Sidebar controls
    with st.sidebar:
        st.subheader("Input")
        _mission = st.selectbox("Mission", ["Auto", "Kepler", "TESS"], index=0)
        target = st.text_input("Target (e.g., KIC/TIC ID)", "")
        _max_rows = st.slider("Max rows (fetch)", 1_000, 100_000, 20_000, step=1_000)
        max_period = st.slider("Max period [days]", 1.0, 50.0, 10.0, step=0.5)
        nbins = st.slider("Phase bins", 25, 300, 100, step=5)
        uploaded = st.file_uploader("Or upload CSV (time,flux)", type=["csv"])
        fetch_btn = st.button("Search")

    # Load any existing artifacts
    model, feature_cols, feature_stats, importances = load_artifacts()

    # Data ingestion
    time_arr, flux_arr = None, None
    if uploaded is not None:
        try:
            df_u = pd.read_csv(uploaded)
            if {"time", "flux"}.issubset(df_u.columns):
                time_arr = df_u["time"].to_numpy(dtype=float)
                flux_arr = df_u["flux"].to_numpy(dtype=float)
            else:
                st.error("CSV must contain 'time' and 'flux' columns.")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

    # TODO: integrate real fetcher by target if available in your repo
    # e.g., from utils.fetchers import fetch_lc; then if fetch_btn and target: time_arr, flux_arr = fetch_lc(target, mission, max_rows)
    if time_arr is None and (fetch_btn or target):
        st.info("No CSV provided. Target fetching is not wired; upload a CSV for now.")

    # If nothing provided, show demo data so the UI is populated
    if time_arr is None:
        time_arr = np.linspace(0, 10, 2000)
        flux_arr = (
            1.0
            + 0.001 * np.sin(2 * np.pi * time_arr / 2.0)
            + 5e-4 * np.random.randn(time_arr.size)
        )

    # Run “BLS” (fallback if real one not available)
    bls = simple_bls(time_arr, flux_arr, max_period=max_period)

    # Metrics header
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Best period [d]", f"{bls['period']:.5g}" if np.isfinite(bls["period"]) else "–"
    )
    c2.metric(
        "Duration [d]",
        f"{bls['duration']:.4g}" if np.isfinite(bls["duration"]) else "–",
    )
    depth_ppm = bls["depth"] * 1e6 if np.isfinite(bls["depth"]) else np.nan
    c3.metric(
        "Depth [frac]",
        f"{bls['depth']:.3g}" if np.isfinite(bls["depth"]) else "–",
        help=f"{depth_ppm:.0f} ppm" if np.isfinite(depth_ppm) else None,
    )
    c4.metric("SDE", f"{bls['sde']:.2f}" if np.isfinite(bls["sde"]) else "–")

    # Plots
    st.subheader("Light curve")
    fig = plot_lightcurve_with_transits(time_arr, flux_arr, bls)
    st.pyplot(fig, clear_figure=True, width="stretch")

    if np.isfinite(bls["period"]):
        st.subheader("Phase-folded")
        fig_phase = plot_phase_folded(time_arr, flux_arr, bls, nbins=nbins)
        st.pyplot(fig_phase, clear_figure=True, width="stretch")

    # AI prediction
    st.subheader("AI prediction")
    label, probs = run_inference(model, feature_cols, feature_stats, bls)
    if label is not None:
        st.write("Prediction:", label)
    if probs:
        st.bar_chart(pd.Series(probs))

    with st.expander("Feature importances", expanded=False):
        plot_importances(importances)


if __name__ == "__main__":
    import lightkurve as lk

    search = lk.search_lightcurve("Kepler-10", mission="Kepler")
    print(search)
    lc = search.download_all(timeout=120).stitch()
    print(lc)
    main()

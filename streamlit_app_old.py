import streamlit as st
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Tuple, Optional

st.title("NASA Light Curve Explorer")
st.markdown("### zamexo — Exploring Exoplanets with Real NASA Data")

target = st.sidebar.text_input("Target name (e.g. TRAPPIST-1)", value="TRAPPIST-1")
mission = st.sidebar.selectbox("Mission", ["TESS", "Kepler", "K2"], index=0)
fetch = st.sidebar.button("Fetch light curve")

lc = None


def _to_numpy(arr):
    """Return a plain 1D numpy float64 array from possible astropy/Quantity/masked input."""
    # lightkurve LightCurve.time is an astropy Time; .value gives numpy array of JD days
    if hasattr(arr, "value"):
        arr = arr.value
    # astropy Time returns np.ndarray of object sometimes; ensure float
    arr = np.asarray(arr, dtype=float)
    return arr.ravel()


def prepare_lightcurve_for_plot(lc_obj, max_points: int = 10000) -> Optional[Tuple[np.ndarray, np.ndarray, dict]]:
    """Extract (time, flux) arrays safe for matplotlib.

    Returns (t_plot, y_plot, stats) or None if data invalid.
    """
    try:
        t = _to_numpy(lc_obj.time)
        y = _to_numpy(lc_obj.flux)
    except Exception as e:  # fallback for tuple or unexpected types
        try:
            t, y = lc_obj
            t, y = _to_numpy(t), _to_numpy(y)
        except Exception:
            st.error(f"Failed to interpret light curve object: {e}")
            return None

    if t.size == 0 or y.size == 0:
        st.warning("Light curve has no data points after extraction.")
        return None

    # Remove NaNs / infs
    mask = np.isfinite(t) & np.isfinite(y)
    if not np.all(mask):
        t, y = t[mask], y[mask]
    if t.size == 0:
        st.warning("All points were NaN/inf after cleaning.")
        return None

    # Basic stats BEFORE downsampling
    stats = {
        "n_points": int(t.size),
        "t_min": float(np.nanmin(t)),
        "t_max": float(np.nanmax(t)),
        "span_days": float(np.nanmax(t) - np.nanmin(t)),
        "flux_mean": float(np.nanmean(y)),
        "flux_std": float(np.nanstd(y)),
        "flux_mad": float(np.nanmedian(np.abs(y - np.nanmedian(y)))),
    }

    # Downsample for plotting if needed
    if t.size > max_points:
        idx = np.linspace(0, t.size - 1, max_points).astype(int)
        t_plot, y_plot = t[idx], y[idx]
        stats["downsampled"] = True
        stats["plotted_points"] = int(max_points)
    else:
        t_plot, y_plot = t, y
        stats["downsampled"] = False
        stats["plotted_points"] = int(t.size)

    return t_plot, y_plot, stats

if fetch:
    with st.spinner("Downloading and processing light curve..."):
        try:
            search = lk.search_lightcurve(target, mission=mission)
            st.write(search)
            if len(search) > 0:
                for i, prod in enumerate(search[:3]):  # Try only first 3 products
                    try:
                        lc = prod.download()
                        st.success(f"Downloaded product #{i}: {prod}")
                        break
                    except Exception as e:
                        st.warning(f"Product #{i} failed: {e}")
                if lc is None:
                    st.error("All products failed to download. Try a different target or mission.")
            else:
                st.error("No products found for this target/mission.")
        except Exception as e:
            st.error(f"Data fetch failed: {e}")
            st.info("Try a different target or mission.")

if lc is not None:
    prepared = prepare_lightcurve_for_plot(lc)
    if prepared is not None:
        t_plot, y_plot, stats = prepared
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Points", f"{stats['n_points']}")
        col2.metric("Span (d)", f"{stats['span_days']:.2f}")
        col3.metric("Flux μ", f"{stats['flux_mean']:.3g}")
        col4.metric("Flux σ", f"{stats['flux_std']:.3g}")
        if stats.get("downsampled"):
            st.caption(f"Downsampled to {stats['plotted_points']} points for rendering.")
        fig, ax = plt.subplots()
        ax.plot(t_plot, y_plot, ".", markersize=2)
        ax.set_xlabel("Time [days]")
        ax.set_ylabel("Flux (relative)")
        ax.set_title(f"{target} ({mission})")
        st.pyplot(fig)
else:
    st.info("Enter a target and click 'Fetch light curve' to begin.")

st.header("Or upload your own light curve CSV")
uploaded_file = st.file_uploader("Upload CSV", type=["csv", "txt"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("CSV preview:", df.head())
        cols = df.select_dtypes(include='number').columns
        if len(cols) >= 2:
            t, y = df[cols[0]], df[cols[1]]
            fig, ax = plt.subplots()
            ax.plot(t, y, ".")
            ax.set_xlabel(cols[0])
            ax.set_ylabel(cols[1])
            st.pyplot(fig)
        else:
            st.error("CSV must have at least two numeric columns.")
    except Exception as e:
        st.error(f"CSV upload failed: {e}")

st.sidebar.info("NASA data download may take up to a minute.")



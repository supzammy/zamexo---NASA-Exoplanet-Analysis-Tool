import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightkurve as lk

st.set_page_config(page_title="Legacy Exoplanet Explorer", layout="wide")
st.title("🌌 Legacy Explainable Exoplanet Explorer UI")

st.markdown(
    """
    This is the earlier aesthetic prototype with richer narrative sections.
    Use this page to compare layout/wording while the new `01.py` evolves.
    """
)

with st.sidebar:
    st.header("Input")
    target = st.text_input("Target", "Kepler-10")
    mission = st.selectbox("Mission", ["Auto", "Kepler", "K2", "TESS"], index=0)
    uploaded = st.file_uploader("Upload CSV (time,flux)", type=["csv"])    
    run = st.button("Analyze")
    enable_explain = st.checkbox("Enable SHAP (if available)")

# Data selection
source_label = ""
time_arr = flux_arr = None
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        if {"time","flux"}.issubset(df.columns):
            time_arr = df["time"].to_numpy(float)
            flux_arr = df["flux"].to_numpy(float)
            source_label = f"Upload: {uploaded.name}"
    except Exception as e:
        st.error(f"Upload failed: {e}")

if time_arr is None and run and target.strip():
    missions = [mission] if mission != "Auto" else ["Kepler","K2","TESS"]
    for m in missions:
        try:
            sr = lk.search_lightcurve(target, mission=m)
            if len(sr)==0: continue
            parts=[]
            for prod in sr[:2]:
                try:
                    lc = prod.download()
                    if lc is not None:
                        parts.append(lc)
                except Exception:
                    pass
            if parts:
                lc_full = parts[0]
                if len(parts)>1:
                    lc_full = lc_full.append(parts[1:])
                t = lc_full.time.value if hasattr(lc_full.time,'value') else np.asarray(lc_full.time)
                f = lc_full.flux.value if hasattr(lc_full.flux,'value') else np.asarray(lc_full.flux)
                msk = np.isfinite(t) & np.isfinite(f)
                time_arr, flux_arr = t[msk], f[msk]
                source_label = f"NASA: {target} ({m})"
                break
        except Exception:
            continue

if time_arr is None:
    # fallback synthetic
    rng = np.random.default_rng(123)
    time_arr = np.linspace(0, 8, 2000)
    flux_arr = 1 + 0.001*np.sin(2*np.pi*time_arr/1.9) + 4e-4*rng.standard_normal(time_arr.size)
    source_label = "Synthetic demo"

# Basic stats
mean_f = float(np.nanmean(flux_arr))
std_f = float(np.nanstd(flux_arr))
span = float(time_arr.max()-time_arr.min())

st.subheader("Overview")
mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Mean", f"{mean_f:.4g}")
mc2.metric("Std", f"{std_f:.3g}")
mc3.metric("Span (d)", f"{span:.2f}")
mc4.metric("Points", f"{time_arr.size}")
st.caption(f"Source: {source_label}")

fig, ax = plt.subplots(figsize=(9,3))
ax.plot(time_arr, flux_arr, '.', ms=2, alpha=0.7)
ax.set_xlabel('Time [d]')
ax.set_ylabel('Flux')
ax.grid(alpha=0.3)
fig.tight_layout()
st.pyplot(fig)

st.markdown("---")
st.markdown("### Explainability Placeholder")
st.info("SHAP and feature-level breakdowns appear here in the new UI (01.py). This legacy page is static for comparison.")

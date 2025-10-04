from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="NASA Transit Finder (MVP)", layout="wide")
st.title("NASA Transit Finder (MVP)")

def sample_curve(n: int = 15000, period: float = 7.2, depth: float = 0.002, duration: float = 0.22, noise: float = 0.0015, seed: int = 42):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 27.4, n)
    y = 1.0 + noise * rng.standard_normal(n)
    phase = ((t - 0.5 * period) % period) / period
    in_tx = (phase < duration / period) | (phase > 1 - duration / period)
    y[in_tx] -= depth
    return t.astype("float64"), y.astype("float64")

def acf_period_estimate(t: np.ndarray, y: np.ndarray, max_period: float) -> dict:
    # Fast, dependency-free heuristic period finder
    if t.size < 20: return {"period": np.nan, "t0": float(t.min()), "duration": np.nan, "depth": np.nan}
    tt = t - t.min()
    med = np.nanmedian(y)
    f = (y / (med if np.isfinite(med) and med != 0 else 1.0)) - 1.0
    n = min(f.size, 6000)
    f = f[:n] - np.nanmean(f[:n])
    ac = np.correlate(f, f, mode="full")[n - 1:]
    ac[0] = 0.0
    if not np.any(np.isfinite(ac[1:])): return {"period": np.nan, "t0": float(t.min()), "duration": np.nan, "depth": np.nan}
    idx = int(np.nanargmax(ac[1:]) + 1)
    dt = float(np.median(np.diff(tt[:n])))
    period = float(idx * dt)
    if not np.isfinite(period) or period <= 0 or period > max_period: period = np.nan
    # crude depth estimate
    depth = float(np.percentile(f, 2)) if np.isfinite(period) else np.nan
    duration = float(max(0.05, 0.05 * period)) if np.isfinite(period) else np.nan
    return {"period": period, "t0": float(t.min()), "duration": duration, "depth": depth}

def fold_curve(t: np.ndarray, period: float, t0: float):
    phase = ((t - t0) % period) / period
    order = np.argsort(phase)
    return phase, order

# Sidebar controls
st.sidebar.header("MVP Controls")
npts = st.sidebar.slider("Points", 2000, 100000, 15000, 1000)
true_period = st.sidebar.slider("True period [days]", 1.0, 20.0, 7.2, 0.1)
depth = st.sidebar.slider("Transit depth", 0.0001, 0.01, 0.002, 0.0001)
duration = st.sidebar.slider("Transit duration [days]", 0.05, 1.0, 0.22, 0.01)
noise = st.sidebar.slider("Noise level", 0.0001, 0.01, 0.0015, 0.0001)
max_period = st.sidebar.slider("Max search period [days]", 2.0, 100.0, 20.0, 0.5)
seed = st.sidebar.number_input("Seed", 0, 1_000_000, 42)
regen = st.sidebar.button("Generate")

# State and data
if "mvp_data" not in st.session_state or regen:
    st.session_state["mvp_data"] = sample_curve(npts, true_period, depth, duration, noise, seed)

t, y = st.session_state["mvp_data"]
st.success(f"Data ready — points: {len(t)}")

# Period estimate
with st.spinner("Estimating period…"):
    est = acf_period_estimate(t, y, max_period)

st.caption(f"Estimate: P≈{est.get('period', np.nan):.4g} d (true {true_period:.3g}), depth≈{(est.get('depth') or 0):.3g}")

# Plot raw
tx, ty = t, y
if tx.size > 40000:
    step = max(1, tx.size // 40000)
    tx, ty = tx[::step], ty[::step]

fig = go.Figure()
mode = "markers" if tx.size > 30000 else "lines"
trace_kwargs = dict(mode=mode)
if mode == "markers": trace_kwargs["marker"] = dict(size=2, color="#1f77b4")
fig.add_scattergl(x=tx, y=ty, name="Flux", **trace_kwargs)

# Overlay estimated transits
if np.isfinite(est.get("period", np.nan)):
    p = float(est["period"]); t0 = float(est["t0"])
    tmin, tmax = float(np.nanmin(t)), float(np.nanmax(t))
    k0 = int(np.floor((tmin - t0) / p)) - 1
    k1 = int(np.ceil((tmax - t0) / p)) + 1
    for k in range(k0, k1 + 1):
        x = t0 + k * p
        if tmin <= x <= tmax:
            fig.add_vline(x=x, line_width=1, line_color="crimson", opacity=0.25)

fig.update_layout(height=380, margin=dict(l=30, r=10, t=10, b=40), xaxis_title="Time [days]", yaxis_title="Flux")
st.subheader("Light curve")
st.plotly_chart(fig, use_container_width=True)

# Folded plot
if np.isfinite(est.get("period", np.nan)) and est["period"] > 0:
    st.subheader("Phase-folded")
    phase, order = fold_curve(t, float(est["period"]), float(est["t0"]))
    ph = phase[order]; fy = y[order]
    nb = 120
    bins = np.linspace(0, 1, nb + 1)
    idx = np.digitize(ph, bins) - 1
    yb = np.array([np.nanmedian(fy[idx == i]) if np.any(idx == i) else np.nan for i in range(nb)])
    xb = 0.5 * (bins[:-1] + bins[1:])
    fp = go.Figure()
    fp.add_scattergl(x=ph, y=fy, mode="markers", marker=dict(size=2, color="#2ca02c"), name="folded")
    fp.add_scatter(x=xb, y=yb, mode="lines", line=dict(color="black", width=1.5), name="binned")
    fp.update_layout(height=360, margin=dict(l=30, r=10, t=10, b=40), xaxis_title="Phase", yaxis_title="Flux")
    st.plotly_chart(fp, use_container_width=True)
else:
    st.info("No reliable period estimated yet. Adjust sliders and click Generate.")
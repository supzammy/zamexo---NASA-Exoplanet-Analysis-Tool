from __future__ import annotations
import numpy as np
import plotly.graph_objects as go

def plot_light_curve(t: np.ndarray, y: np.ndarray, title: str = "Light curve") -> go.Figure:
    fig = go.Figure()
    mode = "markers" if t.size > 30000 else "lines"
    kwargs = dict(mode=mode)
    if mode == "markers":
        kwargs["marker"] = dict(size=2)
    fig.add_scattergl(x=t, y=y, name="flux", **kwargs)
    fig.update_layout(title=title, xaxis_title="Time [days]", yaxis_title="Flux")
    return fig

def plot_folded(phase: np.ndarray, flux: np.ndarray, bins: int = 120) -> go.Figure:
    idx = np.argsort(phase)
    ph = phase[idx]; fl = flux[idx]
    bins_edges = np.linspace(0, 1, bins + 1)
    which = np.digitize(ph, bins_edges) - 1
    yb = np.array([np.nanmedian(fl[which == i]) if np.any(which == i) else np.nan for i in range(bins)])
    xb = 0.5 * (bins_edges[:-1] + bins_edges[1:])
    fig = go.Figure()
    fig.add_scattergl(x=ph, y=fl, mode="markers", marker=dict(size=2), name="folded")
    fig.add_scatter(x=xb, y=yb, mode="lines", line=dict(color="black"), name="binned")
    fig.update_layout(xaxis_title="Phase", yaxis_title="Flux")
    return fig

def plot_feature_importance(names: list[str], importances: np.ndarray) -> go.Figure:
    fig = go.Figure(go.Bar(x=importances, y=names, orientation="h"))
    fig.update_layout(title="Feature importance")
    return fig
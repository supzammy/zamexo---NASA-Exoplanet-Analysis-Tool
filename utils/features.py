from __future__ import annotations

import numpy as np

FEATURE_COLS = [
    "period",
    "duration",
    "depth",
    "sde",
    "t0",
    "depth_ppm",
]

def clean_koi(df):
    out = df.copy()
    rename_map = {
        "koi_period": "period",
        "koi_duration": "duration",
        "koi_depth": "depth",
        "koi_prad": "prad",
        "koi_steff": "steff",
        "koi_slogg": "slogg",
        "koi_srad": "srad",
    }
    out = out.rename(columns=rename_map)
    for k in FEATURE_COLS:
        if k not in out.columns:
            out[k] = np.nan
    return out

def koi_row_for_target(df, target: str) -> dict:
    # Find row for target, else return default
    mask = (df.get("kepler_name") == target) if "kepler_name" in df else None
    if mask is not None and np.any(mask):
        row = df.loc[mask].iloc[0].to_dict()
        for k in FEATURE_COLS:
            row.setdefault(k, np.nan)
        return row
    out = {k: np.nan for k in FEATURE_COLS}
    out["target"] = target
    return out

def map_bls_to_feature_row(cols, bls: dict, fill_stats=None):
    out = {}
    for c in cols:
        if c == "depth_ppm":
            d = bls.get("depth", np.nan)
            out[c] = float(d) * 1e6 if np.isfinite(d) else np.nan
        else:
            out[c] = float(bls.get(c, np.nan)) if c in bls else np.nan
    if fill_stats:
        out.update(fill_stats)
    return out




import json
from pathlib import Path
from typing import Iterable

import numpy as np
from astropy.timeseries import BoxLeastSquares
from .perf import timer


def _clean(time: np.ndarray, flux: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(time, dtype=float)
    f = np.asarray(flux, dtype=float)
    m = np.isfinite(t) & np.isfinite(f)
    t = t[m]
    f = f[m]
    med = np.nanmedian(f)
    if np.isfinite(med) and med != 0:
        f = f / med
    return t, f


def _period_grid(t: np.ndarray, q_min: float, q_max: float, p_min: float, p_max: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build period and duration grids. Uses log grid for periods to spread long periods better.
    """
    periods = np.exp(np.linspace(np.log(p_min), np.log(p_max), int(n)))
    # Map duty cycle q into duration via duration = q * P; scan across q range
    # Use a few representative durations per period; here just mid duty cycle to keep it fast
    q_mid = 0.5 * (q_min + q_max)
    durations = q_mid * periods
    return periods, durations


def bls_features(time: np.ndarray, flux: np.ndarray, max_period: float = 50.0) -> dict:
    t, f = _clean(time, flux)
    if t.size < 50 or np.ptp(t) <= 0:
        return {"period": np.nan, "duration": np.nan, "depth": np.nan, "sde": np.nan, "t0": np.nan}

    min_period = max(0.2, 2.0 * np.median(np.diff(np.sort(t))))
    max_period = float(max_period)
    if min_period >= max_period:
        max_period = min_period * 2.0

    n_periods = int(min(5000, max(500, t.size * 5)))
    periods = np.linspace(min_period, max_period, n_periods)
    durations = 0.02 * periods  # was 0.05 * periods

    bls = BoxLeastSquares(t, f)
    res = bls.power(periods, durations, objective="snr")

    i = int(np.nanargmax(res.power))
    best_period = float(periods[i])
    best_duration = float(durations[i])
    best_t0 = float(res.transit_time[i])
    best_depth = float(getattr(res, "depth", np.full_like(res.power, np.nan))[i])

    p = np.asarray(res.power, dtype=float)
    p_med = np.nanmedian(p)
    p_std = np.nanstd(p)
    sde = float((p[i] - p_med) / p_std) if p_std > 0 else np.nan

    return {"period": best_period, "duration": best_duration, "depth": best_depth, "sde": sde, "t0": best_t0}


def bls_features_fast(time: np.ndarray, flux: np.ndarray, max_period: float = 50.0) -> dict:
    """
    Two-stage BLS: coarse scan then local refinement around peak. Much faster than a single dense sweep.
    """
    t, f = _clean(time, flux)
    if t.size < 50 or np.ptp(t) <= 0:
        return {"period": np.nan, "duration": np.nan, "depth": np.nan, "sde": np.nan, "t0": np.nan}

    T = float(np.ptp(t))
    # Guardrails
    p_min = max(0.2, 2.0 * np.median(np.diff(np.sort(t))))
    p_max = float(max_period)
    if p_min >= p_max:
        p_max = p_min * 2.0

    # Duty cycle bounds (transits often ~0.5%–10%)
    q_min, q_max = 0.005, 0.10

    # Coarse grid size scales with baseline; cap to 1500
    n_coarse = int(min(1500, max(300, T * 20)))
    # Refine grid is small (focused window), ~1200
    n_refine = 1200

    bls = BoxLeastSquares(t, f)
    with timer("BLS coarse"):
        p1, d1 = _period_grid(t, q_min, q_max, p_min, p_max, n_coarse)
        res1 = bls.power(p1, d1, objective="snr")
    i1 = int(np.nanargmax(res1.power))
    p_peak = float(p1[i1])

    # Define narrow refine window around peak (±7%)
    p_lo = max(p_min, p_peak * 0.93)
    p_hi = min(p_max, p_peak * 1.07)
    if p_hi <= p_lo:
        p_lo, p_hi = max(p_min, p_peak * 0.95), min(p_max, p_peak * 1.05)

    with timer("BLS refine"):
        p2, d2 = _period_grid(t, q_min, q_max, p_lo, p_hi, n_refine)
        res2 = bls.power(p2, d2, objective="snr")

    j = int(np.nanargmax(res2.power))
    best_period = float(p2[j])
    best_duration = float(d2[j])
    best_t0 = float(res2.transit_time[j])
    best_depth = float(getattr(res2, "depth", np.full_like(res2.power, np.nan))[j])

    pwr = np.asarray(res2.power, dtype=float)
    p_med = np.nanmedian(pwr)
    p_std = np.nanstd(pwr)
    sde = float((pwr[j] - p_med) / p_std) if p_std > 0 else np.nan

    return {"period": best_period, "duration": best_duration, "depth": best_depth, "sde": sde, "t0": best_t0}


def tls_features(time: np.ndarray, flux: np.ndarray, max_period: float = 50.0) -> dict:
    try:
        from transitleastsquares import transitleastsquares
    except Exception as e:
        raise ImportError("transitleastsquares not installed") from e

    t, f = _clean(time, flux)
    if t.size < 50 or np.ptp(t) <= 0:
        return {"period": np.nan, "duration": np.nan, "depth": np.nan, "sde": np.nan, "t0": np.nan}

    model = transitleastsquares(t, f)
    res = model.power(period_max=float(max_period))
    return {
        "period": float(res.period),
        "duration": float(res.duration),
        "depth": float(res.depth),
        "sde": float(getattr(res, "SDE", np.nan)),
        "t0": float(res.T0 if hasattr(res, "T0") else (t.min())),
    }


def odd_even_test(time: np.ndarray, flux: np.ndarray, period: float, t0: float, duration: float) -> dict:
    t = np.asarray(time, dtype=float)
    f = np.asarray(flux, dtype=float)
    if not (np.isfinite(period) and np.isfinite(t0) and np.isfinite(duration) and period > 0 and duration > 0):
        return {"odd_depth": np.nan, "even_depth": np.nan, "delta_ppm": np.nan, "n_odd": 0, "n_even": 0}

    phase = ((t - t0 + 0.5 * period) % period) - 0.5 * period
    in_tx = np.abs(phase) <= (duration / 2.0)

    k = np.floor((t - t0) / period).astype(int)
    odd_mask = in_tx & ((k % 2) != 0)
    even_mask = in_tx & ((k % 2) == 0)

    def depth_of(mask):
        if mask.sum() < 5:
            return np.nan
        med = np.nanmedian(f[mask])
        return float(1.0 - med)  # depth as fractional drop

    odd_d = depth_of(odd_mask)
    even_d = depth_of(even_mask)
    delta_ppm = float((odd_d - even_d) * 1e6) if np.isfinite(odd_d) and np.isfinite(even_d) else np.nan
    return {"odd_depth": odd_d, "even_depth": even_d, "delta_ppm": delta_ppm, "n_odd": int(odd_mask.sum()), "n_even": int(even_mask.sum())}


def fold_curve(time: np.ndarray, period: float, t0: float) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(time, dtype=float)
    if not (np.isfinite(period) and np.isfinite(t0) and period > 0):
        return np.zeros_like(t), np.arange(t.size)
    phase = ((t - t0) % period) / period
    # Shift to range [-0.5, 0.5) for easier transit centering
    phase[phase >= 0.5] -= 1.0
    order = np.argsort(phase)
    return phase, order


def fetch_toi_table(cache_path=None, force=False):
    import pandas as pd
    return pd.DataFrame()

def fetch_k2_table(cache_path=None, force=False):
    import pandas as pd
    return pd.DataFrame()

OUT_DIR = "models"


def run_bls(time: np.ndarray, flux: np.ndarray, fast: bool = True, max_period: float = 50.0) -> dict:
    """Compatibility wrapper expected by tests; returns dict of BLS features.

    Parameters
    ----------
    time, flux : array-like
        Light curve arrays.
    fast : bool
        If True use two-stage fast BLS, else fallback to standard dense scan.
    max_period : float
        Maximum period to search (days).
    """
    if fast:
        return bls_features_fast(time, flux, max_period=max_period)
    return bls_features(time, flux, max_period=max_period)

"""Shared utility functions for Streamlit pages."""

import numpy as np
import pandas as pd


def _clean_series(arr):
    """Convert to numpy array and handle bad values."""
    try:
        if hasattr(arr, 'value'):  # astropy Quantity
            arr = arr.value
        arr = np.asarray(arr, dtype=float)
        return arr
    except Exception:
        return np.array([])


def summarize_lightcurve(time, flux):
    t = _clean_series(time)
    f = _clean_series(flux)
    m = np.isfinite(t) & np.isfinite(f)
    t, f = t[m], f[m]
    
    if t.size < 10:
        return {
            'n_points': 0, 
            'baseline_days': 0, 
            'cadence_days': np.nan,
            'frac_rms': np.nan, 
            'outlier_fraction': np.nan, 
            'largest_gap_days': np.nan
        }
    
    baseline = float(t.max() - t.min())
    cadence = float(np.nanmedian(np.diff(t)))
    
    flux_med = np.nanmedian(f)
    flux_std = np.nanstd(f)
    rms = flux_std / flux_med if flux_med != 0 else np.nan
    
    # Outlier detection
    thresh = 3 * flux_std
    outliers = np.abs(f - flux_med) > thresh
    outlier_frac = outliers.sum() / len(f)
    
    # Gap analysis
    dt = np.diff(t)
    largest_gap = float(np.nanmax(dt)) if len(dt) > 0 else 0.0
    
    return {
        'n_points': len(t),
        'baseline_days': baseline,
        'cadence_days': cadence,
        'frac_rms': rms,
        'outlier_fraction': outlier_frac,
        'largest_gap_days': largest_gap
    }


import lightkurve as lk

def professional_bls(time, flux, max_period):
    """
    Performs a professional Box Least Squares (BLS) search using lightkurve.
    """
    try:
        # Clean and prepare the light curve object
        t = _clean_series(time)
        f = _clean_series(flux)
        m = np.isfinite(t) & np.isfinite(f)
        t, f = t[m], f[m]

        if t.size < 10:
            return {'period': np.nan, 'duration': np.nan, 'depth': np.nan, 'sde': np.nan, 't0': np.nan}

        # Create a lightkurve object and normalize
        lc = lk.LightCurve(time=t, flux=f).remove_outliers(sigma=5).normalize()

        # Perform the BLS search
        bls = lc.to_periodogram(method='bls', period=np.arange(0.5, max_period, 0.01))
        
        # Extract results
        period = bls.period_at_max_power.value
        duration = bls.duration_at_max_power.value
        t0 = bls.transit_time_at_max_power.value
        depth = bls.depth_at_max_power
        sde = bls.max_power
        
        # Validate results
        if not all(np.isfinite([period, duration, t0, depth, sde])):
            raise ValueError("BLS returned non-finite values")

        return {
            'period': period,
            'duration': duration,
            'depth': depth,
            'sde': sde,
            't0': t0
        }
    except Exception as e:
        # Fallback to a very basic estimate if professional BLS fails
        return {'period': np.nan, 'duration': np.nan, 'depth': np.nan, 'sde': np.nan, 't0': np.nan}
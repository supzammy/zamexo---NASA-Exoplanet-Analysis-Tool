from __future__ import annotations
import numpy as np
from utils.features import run_bls, fold_curve, bls_features
from nasa_project.features import extract_basic
from typing import Optional, Tuple

FEATURE_COLS = [
    "period",
    "duration",
    "depth",
    "sde",
    "t0",
    "depth_ppm",
]

def clean_koi(df):
    try:
        return df.copy()
    except Exception:
        return df

def koi_row_for_target(target: str) -> dict:
    row = {k: np.nan for k in FEATURE_COLS}
    row["target"] = target
    return row

def map_bls_to_feature_row(cols, bls: dict):
    out = {}
    for c in cols:
        if c == "depth_ppm":
            d = bls.get("depth", np.nan)
            out[c] = float(d) * 1e6 if np.isfinite(d) else np.nan
        else:
            out[c] = float(bls.get(c, np.nan)) if c in bls else np.nan
    return out

def test_extract_basic_nan():
    # All flux values are nan
    t = np.arange(5)
    f = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
    bf = extract_basic(t, f)
    assert bf.n == 0
    assert np.isnan(bf.mean)
    assert np.isnan(bf.std)
    assert np.isnan(bf.min)
    assert np.isnan(bf.max)

def test_extract_basic_inf():
    # Some flux values are inf
    t = np.arange(5)
    f = np.array([1.0, np.inf, 0.9, -np.inf, 1.05])
    bf = extract_basic(t, f)
    assert bf.n == 3
    assert 0.9 <= bf.min <= 1.05
    assert 0.9 <= bf.max <= 1.05

def test_extract_basic_empty():
    # Empty flux array
    t = np.array([])
    f = np.array([])
    bf = extract_basic(t, f)
    assert bf.n == 0
    assert np.isnan(bf.mean)
    assert np.isnan(bf.std)
    assert np.isnan(bf.min)
    assert np.isnan(bf.max)

def test_extract_basic_single_value():
    # Single value
    t = np.array([0])
    f = np.array([1.23])
    bf = extract_basic(t, f)
    assert bf.n == 1
    assert bf.mean == 1.23
    assert bf.std == 0.0
    assert bf.min == 1.23
    assert bf.max == 1.23

def test_extract_basic_mixed_finite():
    # Mixed finite and non-finite
    t = np.arange(6)
    f = np.array([1.0, np.nan, 2.0, np.inf, 3.0, -np.inf])
    bf = extract_basic(t, f)
    assert bf.n == 3
    assert bf.min == 1.0
    assert bf.max == 3.0
    assert np.isclose(bf.mean, 2.0)
    assert bf.std > 0

def test_extract_basic_large_array():
    # Large random array
    rng = np.random.default_rng(123)
    t = np.arange(10000)
    f = rng.normal(1.0, 0.1, size=10000)
    bf = extract_basic(t, f)
    assert bf.n == 10000
    assert abs(bf.mean - 1.0) < 0.01
    assert 0.09 < bf.std < 0.11
    assert bf.min < bf.max

def test_extract_basic_all_zeros():
    # All zeros
    t = np.arange(10)
    f = np.zeros(10)
    bf = extract_basic(t, f)
    assert bf.n == 10
    assert bf.mean == 0.0
    assert bf.std == 0.0
    assert bf.min == 0.0
    assert bf.max == 0.0

def test_extract_basic_negative_values():
    # Negative values
    t = np.arange(5)
    f = np.array([-1.0, -2.0, -3.0, -4.0, -5.0])
    bf = extract_basic(t, f)
    assert bf.n == 5
    assert bf.min == -5.0
    assert bf.max == -1.0
    assert bf.mean < 0
    assert bf.std > 0

def test_extract_basic_mixed_sign():
    # Mixed positive and negative
    t = np.arange(6)
    f = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, np.nan])
    bf = extract_basic(t, f)
    assert bf.n == 5
    assert bf.min == -2.0
    assert bf.max == 2.0
    assert np.isclose(bf.mean, 0.0)
    assert bf.std > 0

def test_extract_basic_with_large_and_small():
    # Large and small values
    t = np.arange(4)
    f = np.array([1e-10, 1e10, -1e10, 0.0])
    bf = extract_basic(t, f)
    assert bf.n == 4
    assert bf.min == -1e10
    assert bf.max == 1e10
    assert bf.std > 0

def test_extract_basic_with_masked_array():
    # Masked array support
    t = np.arange(5)
    f = np.ma.array([1.0, 2.0, 3.0, 4.0, 5.0], mask=[0, 1, 0, 1, 0])
    bf = extract_basic(t, f)
    # Only unmasked: 1.0, 3.0, 5.0
    assert bf.n == 3
    assert bf.min == 1.0
    assert bf.max == 5.0
    assert np.isclose(bf.mean, 3.0)

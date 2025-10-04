"""
Lightweight tests for the project (pytest).
Run: make test
"""

import math
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pytest

features = pytest.importorskip("utils.features")


def _period_key(d: Dict[str, Any]) -> Optional[str]:
    for k in ("koi_period", "period", "bls_period", "best_period"):
        if k in d:
            return k
    return None


def _norm_local(s: str) -> str:
    return s.lower().replace(" ", "").replace("-", "")


def test_core_imports():
    import astropy
    import lightkurve
    import matplotlib

    assert astropy is not None and lightkurve is not None and matplotlib is not None
    assert isinstance(np.__version__, str)
    assert isinstance(pd.__version__, str)


def test_feature_columns_unique_and_nonempty():
    cols = features.FEATURE_COLS
    assert isinstance(cols, list) and len(cols) > 0
    assert len(cols) == len(set(cols))
    assert all(isinstance(c, str) and c for c in cols)


def test_clean_koi_roundtrip_subset_columns():
    df = pd.DataFrame(
        [
            {
                "koi_period": 7.1,
                "koi_duration": 2.2,
                "koi_depth": 150.0,
                "koi_srad": 0.95,
                "koi_disposition": "CANDIDATE",
                "kepid": 42,
                "kepoi_name": "KOI-42.01",
                "kepler_name": "Kepler-42 b",
            }
        ]
    )
    cleaned = features.clean_koi(df)
    keep_cols = set(
        features.FEATURE_COLS
        + [features.LABEL_COL, "kepid", "kepoi_name", "kepler_name"]
    )
    for col in keep_cols.intersection(df.columns):
        assert cleaned.iloc[0][col] == df.iloc[0][col]


def test_bls_features_on_synthetic_transit():
    if not hasattr(features, "bls_features"):
        pytest.skip("bls_features not available")
    rng = np.random.default_rng(42)
    p_true, width, depth = 2.5, 0.12, 0.008
    time = np.linspace(0, 30, 3000)
    phase = np.mod(time, p_true)
    in_tr = (phase < width / 2.0) | (phase > (p_true - width / 2.0))
    flux = np.ones_like(time)
    flux[in_tr] -= depth
    flux += rng.normal(0.0008, 0.0008, size=flux.size)
    out = features.bls_features(time, flux)
    assert isinstance(out, dict) and out
    pkey = _period_key(out)
    assert pkey is not None
    pest = float(out[pkey])
    rel = min(abs(pest - p_true) / p_true, abs(pest - p_true / 2) / (p_true / 2))
    assert rel < 0.15
    for k in ("duration", "depth", "sde"):
        if k in out:
            assert math.isfinite(float(out[k]))

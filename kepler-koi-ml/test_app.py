"""
Lightweight sanity tests and utils.features tests (pytest).
"""

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest


def _norm_local(s: str) -> str:
    return s.lower().replace(" ", "").replace("-", "")


def test_basic_functionality():
    rng = np.random.default_rng(0)
    time = np.linspace(0, 10, 200)
    flux = 1 + 0.01 * np.sin(time * 2)
    flux[80:120] -= 0.015
    flux += rng.normal(0, 0.001, size=time.size)
    assert len(time) == 200
    assert len(flux) == 200
    assert float(np.min(flux)) < 0.99


def test_plot_creation():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time = np.linspace(0, 10, 200)
    flux = 1 + 0.01 * np.sin(time * 2)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, flux)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Relative Brightness")
    assert ax.get_xlabel() == "Time (days)"
    assert ax.get_ylabel() == "Relative Brightness"


def test_core_imports():
    import astropy
    import lightkurve
    import matplotlib

    _ = (
        getattr(astropy, "__version__", ""),
        getattr(lightkurve, "__version__", ""),
        getattr(matplotlib, "__version__", ""),
    )
    assert True


# ---- utils.features focused tests ----

features = pytest.importorskip("utils.features")


def _period_key(d: dict[str, Any]) -> str | None:
    for k in ("koi_period", "period", "bls_period", "best_period", "candidate_period"):
        if k in d:
            return k
    return None


def _duration_key(d: dict[str, Any]) -> str | None:
    for k in ("koi_duration", "duration", "bls_duration", "best_duration"):
        if k in d:
            return k
    return None


def _depth_key(d: dict[str, Any]) -> str | None:
    for k in ("koi_depth", "depth", "bls_depth", "best_depth"):
        if k in d:
            return k
    return None


def test_feature_columns_unique_and_nonempty():
    cols = features.FEATURE_COLS
    assert isinstance(cols, list) and len(cols) > 0
    assert len(cols) == len(set(cols))
    assert all(isinstance(c, str) and c for c in cols)


def test_clean_koi_handles_missing_expected_columns_gracefully():
    df = pd.DataFrame(
        [
            {"koi_period": 5.0, "koi_disposition": "CANDIDATE", "kepid": 1},
            {"koi_duration": 2.1, "koi_disposition": "FALSE POSITIVE", "kepid": 2},
        ]
    )
    cleaned = features.clean_koi(df)
    for keep in ("kepid", features.LABEL_COL):
        assert keep in cleaned.columns
    for col in set(features.FEATURE_COLS).intersection(df.columns):
        assert col in cleaned.columns
    assert len(cleaned) == len(df)


def test_koi_row_for_target_is_case_and_symbol_insensitive():
    df = pd.DataFrame(
        [
            {"kepler_name": "Kepler-22", "kepoi_name": "KOI-0871.01"},
            {"kepler_name": "HD 209458", "kepoi_name": "KOI-0001.01"},
        ]
    )
    a = features.koi_row_for_target(df, "kepler 22")
    b = features.koi_row_for_target(df, "Kepler-22")
    c = features.koi_row_for_target(df, "HD-209458")
    d = features.koi_row_for_target(df, "hd 209458")
    assert a is not None and b is not None and c is not None and d is not None
    assert a["kepler_name"] == b["kepler_name"] == "Kepler-22"
    assert c["kepler_name"] == d["kepler_name"] == "HD 209458"
    assert features.koi_row_for_target(df, "TRAPPIST-1") is None


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Kepler-10", "kepler10"),
        ("Kepler 10", "kepler10"),
        ("TRAPPIST-1", "trappist1"),
        ("HD 209458", "hd209458"),
        ("Kepler-186", "kepler186"),
        ("  Kepler - 186  ", "kepler186"),
    ],
)
def test_optional_norm_name_if_exposed(raw: str, expected: str):
    if not hasattr(features, "norm_name"):
        pytest.skip("utils.features.norm_name not available")
    got = features.norm_name(raw)
    assert got == expected
    assert got == _norm_local(raw)


def test_bls_features_recovers_period_on_synthetic_transit():
    if not hasattr(features, "bls_features"):
        pytest.skip("utils.features.bls_features not available")

    rng = np.random.default_rng(42)
    p_true = 2.5
    width = 0.12
    depth = 0.008
    time = np.linspace(0, 30, 3000)
    phase = np.mod(time, p_true)
    in_transit = (phase < width / 2.0) | (phase > (p_true - width / 2.0))
    flux = np.ones_like(time)
    flux[in_transit] -= depth
    flux += rng.normal(0.0008, 0.0008, size=flux.size)

    result = features.bls_features(time, flux)
    assert isinstance(result, dict) and len(result) > 0

    p_key = _period_key(result)
    assert p_key is not None
    p_est = float(result[p_key])

    rel_err = min(abs(p_est - p_true) / p_true, abs(p_est - p_true / 2.0) / (p_true / 2.0))
    assert rel_err < 0.15

    d_key = _depth_key(result)
    if d_key is not None:
        assert result[d_key] > 0

    w_key = _duration_key(result)
    if w_key is not None:
        assert result[w_key] > 0


def test_bls_features_accepts_minimal_input_and_returns_numbers():
    if not hasattr(features, "bls_features"):
        pytest.skip("utils.features.bls_features not available")
    rng = np.random.default_rng(0)
    time = np.linspace(0, 10, 1000)
    flux = 1.0 + rng.normal(0.0005, 0.0005, size=time.size)
    out = features.bls_features(time, flux)
    assert isinstance(out, dict) and out
    for _, v in out.items():
        if isinstance(v, (int, float, np.floating)):
            assert math.isfinite(float(v))


# Cross-module norm check
_nasa = pytest.importorskip("utils.nasa")


def test_norm_name_consistency_across_modules_when_available():
    names = ["Kepler-10", "Kepler 10", "TRAPPIST-1", "HD 209458", "Kepler-186"]
    f_norm = getattr(features, "norm_name", None)
    n_norm = getattr(_nasa, "norm_name", None)
    if f_norm is None and n_norm is None:
        pytest.skip("No norm_name in either module")
    for raw in names:
        local = _norm_local(raw)
        if f_norm:
            assert f_norm(raw) == local
        if n_norm:
            assert n_norm(raw) == local


def test_sanity():
    assert True

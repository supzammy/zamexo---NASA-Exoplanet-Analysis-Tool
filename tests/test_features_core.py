import numpy as np

from utils.features import bls_features, fold_curve


def test_bls_on_synthetic():
    # Synthetic light curve with a transit-like dip
    time = np.linspace(0, 10, 2000)
    flux = 1.0 + 0.0005 * np.random.randn(time.size)
    flux[(time % 2.5) < 0.15] -= 0.01  # period≈2.5d, duration≈0.15d
    feats = bls_features(time, flux, max_period=5.0)
    print(feats)
    phase, order = fold_curve(time, period=feats["period"], t0=feats["t0"])


def test_fold_curve_basic():
    t = np.linspace(0, 10, 1000)
    phase, order = fold_curve(t, period=2.5, t0=0.0)
    assert phase.shape == t.shape
    assert order.shape == t.shape
    assert np.all(np.isfinite(phase))
    assert phase.min() >= -0.5 - 1e-12
    assert phase.max() <= 0.5 + 1e-12
    assert np.all(np.diff(phase) >= -1e-12)

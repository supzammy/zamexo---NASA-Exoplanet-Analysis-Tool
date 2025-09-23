import numpy as np

from utils.features import bls_features


def test_bls_on_synthetic():
    # Make a synthetic light curve with a dip (transit-like)
    t = np.linspace(0, 10, 2000)
    y = 1.0 + 0.0005 * np.random.randn(t.size)
    mask = (t % 2.5) < 0.15  # period≈2.5d, duration≈0.15d
    y[mask] -= 0.01
    feats = bls_features(t, y, max_period=5.0)
    assert feats["sde"] > 5.0
    assert feats["depth"] < 0.0  # negative dip

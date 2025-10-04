# makes 'tests' a Python package

import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.nasa import fetch_light_curve
from utils.features import bls_features
from scripts.train_baseline import main as train_main
from tests.test_features import test_bls_on_synthetic
from tests.test_plots import test_plot_labels
from tests.test_training import test_training_artifacts


def test_bls_on_synthetic():
    # Synthetic light curve with a transit-like dip
    t = np.linspace(0, 10, 2000)
    y = 1.0 + 0.0005 * np.random.randn(t.size)
    mask = (t % 2.5) < 0.15  # period≈2.5d, duration≈0.15d
    y[mask] -= 0.01
    feats = bls_features(t, y, max_period=5.0)
    assert feats["sde"] > 5.0
    assert feats["depth"] < 0.0  # negative dip


def test_plot_labels():
    t = np.linspace(0, 10, 200)
    y = 1 + 0.01 * np.sin(t)
    fig, ax = plt.subplots()
    ax.plot(t, y)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Relative Brightness")
    assert ax.get_xlabel() == "Time (days)"
    assert ax.get_ylabel() == "Relative Brightness"


def test_training_artifacts(tmp_path: Path = Path("models")):
    acc = train_main()
    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "feature_cols.json").exists()
    assert (tmp_path / "metrics.json").exists()
    assert acc > 0.7


def main():
    test_bls_on_synthetic()
    test_plot_labels()
    test_training_artifacts()
    print("✅ all tests passed")


if __name__ == "__main__":
    # Test fetching a real light curve
    time, flux = fetch_light_curve("Kepler-10", mission="Kepler")
    if time is not None:
        print("Fetched points:", len(time))
    else:
        print("Fetch failed.")

    # Run all tests
    main()

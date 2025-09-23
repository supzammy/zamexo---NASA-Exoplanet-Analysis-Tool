import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def test_plot_labels():
    t = np.linspace(0, 10, 200)
    y = 1 + 0.01 * np.sin(t)
    fig, ax = plt.subplots()
    ax.plot(t, y)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Relative Brightness")
    assert ax.get_xlabel() == "Time (days)"
    assert ax.get_ylabel() == "Relative Brightness"

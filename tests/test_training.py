import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path

import numpy as np
import pandas as pd

from scripts import train_baseline
from utils import nasa as nasa_mod


def test_training_artifacts(monkeypatch, tmp_path: Path):
    # Offline stub for KOI table to avoid network
    n = 60
    df = pd.DataFrame(
        {
            "koi_period": np.random.uniform(1, 10, n),
            "koi_duration": np.random.uniform(0.05, 0.5, n),
            "koi_depth": np.random.uniform(50, 1000, n),  # ppm
            "koi_prad": np.random.uniform(0.5, 5.0, n),
            "koi_steff": np.random.uniform(4800, 6200, n),
            "koi_slogg": np.random.uniform(4.0, 4.6, n),
            "koi_srad": np.random.uniform(0.7, 1.5, n),
            "koi_disposition": np.random.choice(
                ["FALSE POSITIVE", "CANDIDATE", "CONFIRMED"], n
            ),
        }
    )
    monkeypatch.setattr(
        nasa_mod, "fetch_koi_table", lambda cache_path="x", force=False: df
    )

    # Point models dir to a temp folder for the test run
    out_dir = tmp_path / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Monkeypatch OUT_DIR inside the module
    monkeypatch.setattr(train_baseline, "OUT_DIR", out_dir)

    # Run training with no accuracy gate
    train_baseline.main(force_download=False, min_accuracy=0.0)

    assert (out_dir / "model.joblib").exists()
    assert (out_dir / "feature_cols.json").exists()
    assert (out_dir / "metrics.json").exists()

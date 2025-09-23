import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path

import pandas as pd

from scripts.train_baseline import main as train_main, train


def test_training_artifacts(tmp_path: Path = Path("models")):
    acc = train_main()
    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "feature_cols.json").exists()
    assert (tmp_path / "metrics.json").exists()
    assert acc > 0.7


def test_train_baseline_on_tiny_df():
    df = pd.DataFrame(
        {
            "koi_period": [1.0, 2.0, 3.0, 4.0],
            "koi_duration": [0.1, 0.2, 0.3, 0.4],
            "koi_depth": [100, 200, 150, 300],
            "koi_prad": [1.0, 2.0, 1.5, 2.5],
            "koi_steff": [5700, 5500, 5300, 5200],
            "koi_slogg": [4.4, 4.3, 4.5, 4.2],
            "koi_srad": [1.0, 0.9, 1.1, 1.2],
            "koi_disposition": ["CANDIDATE", "FALSE POSITIVE", "CANDIDATE", "CONFIRMED"],
        }
    )
    pipe, acc, report, cm, cols = train(df)
    assert hasattr(pipe, "predict")
    assert len(cols) == 7

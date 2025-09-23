import pandas as pd

from utils.nasa import fetch_koi_table, koi_row_for_target


def test_koi_row_for_target_mask_alignment():
    df = pd.DataFrame(
        {
            "kepler_name": ["Kepler-10", None, "Kepler-22"],
            "kepoi_name": ["K00001.01", "K00002.01", "K00003.01"],
            "koi_disposition": ["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"],
            "koi_period": [1.0, 2.0, 3.0],
            "koi_duration": [0.1, 0.2, 0.3],
            "koi_depth": [200, 300, 400],
            "koi_prad": [1.1, 2.2, 3.3],
            "koi_steff": [5600, 5700, 5800],
            "koi_slogg": [4.4, 4.3, 4.2],
            "koi_srad": [1.0, 0.9, 1.2],
        }
    )
    row = koi_row_for_target(df, "Kepler-10")
    assert row is not None
    assert str(row["kepoi_name"]) == "K00001.01"


def test_fetch_uses_cache(tmp_path, monkeypatch):
    # Avoid network by pointing fetch to a temp cache
    p = tmp_path / "koi.csv"
    df = pd.DataFrame(
        {
            "koi_period": [1.0],
            "koi_duration": [0.1],
            "koi_depth": [100],
            "koi_prad": [1.0],
            "koi_steff": [5700],
            "koi_slogg": [4.4],
            "koi_srad": [1.0],
            "koi_disposition": ["CANDIDATE"],
        }
    )
    df.to_csv(p, index=False)
    got = fetch_koi_table(cache_path=p, force=False)
    assert list(got.columns) == list(df.columns)
    assert len(got) == 1

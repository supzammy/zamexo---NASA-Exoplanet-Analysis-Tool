import numpy as np

from utils.features import FEATURE_COLS, map_bls_to_feature_row


def test_map_bls_to_feature_row_fills_stats():
    bls = {"period": 3.14, "duration": 0.12, "depth": -0.001}
    fill = {"koi_prad": 1.5, "koi_steff": 5600.0, "koi_slogg": 4.4, "koi_srad": 1.0}
    row = map_bls_to_feature_row(FEATURE_COLS, bls, fill_stats=fill)
    assert row["koi_period"] == bls["period"]
    assert row["koi_duration"] == bls["duration"]
    assert row["koi_depth"] == bls["depth"]
    for k in ["koi_prad", "koi_steff", "koi_slogg", "koi_srad"]:
        assert row[k] == fill[k]


def test_map_bls_to_feature_row_handles_missing():
    row = map_bls_to_feature_row(FEATURE_COLS, bls={}, fill_stats={})
    assert np.isnan(row["koi_period"])
    assert np.isnan(row["koi_duration"])
    assert np.isnan(row["koi_depth"])

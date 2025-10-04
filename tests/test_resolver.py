import logging

import numpy as np
import pandas as pd

from utils import nasa


def test_resolve_features_for_target_offline(monkeypatch):
    # Stub tables
    df_koi = pd.DataFrame(
        {
            "kepler_name": ["Kepler-10"],
            "koi_period": [0.84],
            "koi_duration": [0.1],
            "koi_depth": [200],  # ppm
            "koi_prad": [1.4],
            "koi_steff": [5700],
            "koi_slogg": [4.4],
            "koi_srad": [1.0],
            "koi_disposition": ["CONFIRMED"],
        }
    )
    df_toi = pd.DataFrame(
        {
            "toi": ["TOI 123"],
            "orbital_period": [5.1],
            "transit_duration": [0.12],
            "transit_depth": [500],  # ppm
            "pl_rade": [1.5],
            "st_teff": [5600],
            "st_logg": [4.4],
            "st_rad": [1.0],
            "tfopwg_disp": ["PC"],
        }
    )
    df_k2 = pd.DataFrame(
        {
            "epic": ["201000001"],
            "period": [10.2],
            "duration": [0.2],
            "depth": [800],  # ppm
            "planet_radius": [2.0],
            "st_teff": [5500],
            "st_logg": [4.3],
            "st_rad": [0.9],
            "disposition": ["CANDIDATE"],
        }
    )

    monkeypatch.setattr(
        nasa, "fetch_koi_table", lambda cache_path="unused", force=False: df_koi
    )
    monkeypatch.setattr(
        nasa, "fetch_toi_table", lambda cache_path="unused", force=False: df_toi
    )
    monkeypatch.setattr(
        nasa, "fetch_k2_table", lambda cache_path="unused", force=False: df_k2
    )

    r1 = nasa.resolve_features_for_target("Kepler-10")
    assert r1 and r1["source"] == "KOI" and len(r1["features"]) == 7

    r2 = nasa.resolve_features_for_target("TOI 123")
    assert r2 and r2["source"] == "TOI"
    # depth normalized from 500 ppm -> 0.0005
    assert np.isclose(float(r2["features"][2]), 0.0005, rtol=1e-6)

    r3 = nasa.resolve_features_for_target("EPIC 201000001")
    assert r3 and r3["source"] == "K2"


def test_resolve_features_for_target_toi(monkeypatch):
    toi_df = pd.DataFrame(
        {
            "toi": ["TOI 123"],
            "orbital_period": [5.1],
            "transit_duration": [0.12],
            "transit_depth": [500],  # ppm
            "pl_rade": [1.5],
            "st_teff": [5600],
            "st_logg": [4.4],
            "st_rad": [1.0],
            "tfopwg_disp": ["PC"],
        }
    )
    monkeypatch.setattr(
        nasa, "fetch_toi_table", lambda cache_path="x", force=False: toi_df
    )
    res = nasa.resolve_features_for_target("TOI 123")
    assert res is not None and res["source"] == "TOI"
    feats = res["features"]
    assert isinstance(feats, np.ndarray) and feats.shape == (7,)
    assert abs(float(feats[2]) - 0.0005) < 1e-10


def test_fetch_k2_handles_error_csv(monkeypatch, tmp_path, caplog):
    err_csv = 'ERROR<br>,Message<br>\n"UserError - table parameter","k2candidates is not a valid table"\n'
    monkeypatch.setattr(
        nasa.requests,
        "get",
        lambda url, timeout=60: type(
            "R", (), {"text": err_csv, "raise_for_status": lambda self: None}
        )(),
    )
    p = tmp_path / "k2.csv"
    with caplog.at_level(logging.WARNING, logger="utils.nasa"):
        df = nasa.fetch_k2_table(cache_path=p, force=True)
    assert df.empty

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
            "koi_depth": [200],
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
            "depth": [800],  # assume ppm-like -> normalized inside
            "planet_radius": [2.0],
            "st_teff": [5500],
            "st_logg": [4.3],
            "st_rad": [0.9],
            "disposition": ["CANDIDATE"],
        }
    )

    monkeypatch.setattr(nasa, "fetch_koi_table", lambda cache_path="unused", force=False: df_koi)
    monkeypatch.setattr(nasa, "fetch_toi_table", lambda cache_path="unused", force=False: df_toi)
    monkeypatch.setattr(nasa, "fetch_k2_table", lambda cache_path="unused", force=False: df_k2)

    r1 = nasa.resolve_features_for_target("Kepler-10")
    assert r1 and r1["source"] == "KOI" and len(r1["features"]) == 7

    r2 = nasa.resolve_features_for_target("TOI 123")
    assert r2 and r2["source"] == "TOI"
    # depth normalized from 500 ppm -> 0.0005
    assert np.isclose(r2["features"][2], 0.0005, rtol=1e-6)

    r3 = nasa.resolve_features_for_target("EPIC 201000001")
    assert r3 and r3["source"] == "K2"


def test_resolve_features_for_target_toi(monkeypatch):
    # Arrange: tiny TOI row (depth in ppm) and mock fetch_toi_table
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
    monkeypatch.setattr(nasa, "fetch_toi_table", lambda cache_path="x", force=False: toi_df)

    # Act
    res = nasa.resolve_features_for_target("TOI 123")

    # Assert
    assert res is not None
    assert res["source"] == "TOI"
    feats = res["features"]
    assert isinstance(feats, np.ndarray) and feats.shape == (7,)
    # depth normalized: 500 ppm -> 0.0005
    assert abs(float(feats[2]) - 0.0005) < 1e-10


def test_resolve_features_for_target_koi(monkeypatch):
    # Arrange: tiny KOI row with direct match by kepler_name
    koi_df = pd.DataFrame(
        {
            "kepler_name": ["Kepler-1 b"],
            "kepoi_name": ["K00001.01"],
            "kepid": ["1234567"],
            "koi_period": [1.0],
            "koi_duration": [0.1],
            "koi_depth": [100],  # ppm
            "koi_prad": [1.0],
            "koi_steff": [5700],
            "koi_slogg": [4.4],
            "koi_srad": [1.0],
            "koi_disposition": ["CANDIDATE"],
        }
    )
    monkeypatch.setattr(nasa, "fetch_koi_table", lambda cache_path="x", force=False: koi_df)

    # Act
    res = nasa.resolve_features_for_target("Kepler-1 b")

    # Assert
    assert res is not None
    assert res["source"] == "KOI"
    feats = res["features"]
    # depth normalized: 100 ppm -> 0.0001
    assert abs(float(feats[2]) - 0.0001) < 1e-12

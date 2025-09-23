import io

import pandas as pd


def _fake_resp(csv_text: str):
    class R:
        def __init__(self, t):
            self.text = t

        def raise_for_status(self):
            return None

    return R(csv_text)


def test_fetch_koi_mock(monkeypatch, tmp_path):
    from utils import nasa

    csv = "koi_period,koi_duration,koi_depth,koi_prad,koi_steff,koi_slogg,koi_srad,koi_disposition\n1.0,0.1,100,1.0,5700,4.4,1.0,CANDIDATE\n"
    monkeypatch.setattr(nasa.requests, "get", lambda url, timeout=60: _fake_resp(csv))
    p = tmp_path / "koi.csv"
    df = nasa.fetch_koi_table(cache_path=p, force=True)
    assert len(df) == 1
    assert set(["koi_period", "koi_disposition"]).issubset(df.columns)


def test_fetch_toi_mock(monkeypatch, tmp_path):
    from utils import nasa

    csv = "toi,orbital_period,transit_duration,transit_depth,pl_rade,st_teff,st_logg,st_rad,tfopwg_disp\nTOI 123,5.1,0.12,500,1.5,5600,4.4,1.0,PC\n"
    monkeypatch.setattr(nasa.requests, "get", lambda url, timeout=60: _fake_resp(csv))
    p = tmp_path / "toi.csv"
    df = nasa.fetch_toi_table(cache_path=p, force=True)
    assert len(df) == 1
    assert "toi" in df.columns


def test_fetch_k2_mock(monkeypatch, tmp_path):
    from utils import nasa

    csv = "epic,period,duration,depth,planet_radius,st_teff,st_logg,st_rad,disposition\n201000001,10.2,0.2,800,2.0,5500,4.3,0.9,CANDIDATE\n"
    monkeypatch.setattr(nasa.requests, "get", lambda url, timeout=60: _fake_resp(csv))
    p = tmp_path / "k2.csv"
    df = nasa.fetch_k2_table(cache_path=p, force=True)
    assert len(df) == 1
    assert "epic" in df.columns

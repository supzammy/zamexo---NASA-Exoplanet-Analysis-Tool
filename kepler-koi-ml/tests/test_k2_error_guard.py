import logging

import pandas as pd
import pytest


class _FakeResp:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


def test_fetch_k2_handles_error_csv(monkeypatch, tmp_path, caplog):
    from utils import nasa

    # Simulate NASA error page returned as CSV-like content
    err_csv = (
        'ERROR<br>,Message<br>\n"UserError - table parameter","k2candidates is not a valid table"\n'
    )
    monkeypatch.setattr(nasa.requests, "get", lambda url, timeout=60: _FakeResp(err_csv))
    p = tmp_path / "k2.csv"
    with caplog.at_level(logging.WARNING, logger="utils.nasa"):
        df = nasa.fetch_k2_table(cache_path=p, force=True)
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert "K2 API returned an error page" in caplog.text

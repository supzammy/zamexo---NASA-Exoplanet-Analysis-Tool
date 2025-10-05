"""
NASA Exoplanet Archive utilities for KOI, TOI, K2 table fetching and ephemerides.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import lightkurve as lk
import pandas as pd
import requests
from functools import lru_cache


KOI_TABLE = "q1_q17_dr25_koi"
KOI_URL = f"https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+{KOI_TABLE}+&format=csv"

# Placeholder / representative endpoints (tests monkeypatch requests.get so content is controlled there)
TOI_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+toi&format=csv"
K2_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+k2&format=csv"


def fetch_koi_table(cache_path: str | Path = "data/koi_cache.csv", force: bool = False) -> pd.DataFrame:
    cache = Path(cache_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and not force:
        return pd.read_csv(cache)
    resp = requests.get(KOI_URL, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df.to_csv(cache, index=False)
    return df


def prepare_koi_training(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "koi_disposition",
        "koi_period",
        "koi_duration",
        "koi_depth",
        "koi_prad",
        "koi_steff",
        "koi_slogg",
        "koi_srad",
        "koi_impact",
        "koi_snr",
    ]
    keep = [c for c in cols if c in df.columns]
    out = df[keep].dropna()
    out = out[out["koi_disposition"].isin(["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"])]
    return out.reset_index(drop=True)


def fetch_ephemerides(target: str) -> list[dict]:
    """
    Return list of dicts with name, period (days), t0 (BJD_TDB) for confirmed planets.
    Tries matching by planet name (pl_name) and host star (hostname).
    """
    base = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    cols = "pl_name,hostname,pl_orbper,tranmid"
    q1 = f"select+{cols}+from+pscomppars+where+lower(pl_name)=lower('{target}')"
    q2 = f"select+{cols}+from+pscomppars+where+lower(hostname)=lower('{target}')"
    out: list[dict] = []
    for q in (q1, q2):
        url = f"{base}?query={q}&format=csv"
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            if df.empty:
                continue
            for _, row in df.iterrows():
                p = float(row.get("pl_orbper", float("nan")))
                t0 = float(row.get("tranmid", float("nan")))
                name = str(row.get("pl_name") or row.get("hostname") or target)
                if pd.notna(p):
                    out.append({"name": name, "period": p, "t0": t0})
        except Exception as e:
            logging.warning(f"Failed to fetch ephemerides for {target}: {e}")
            continue
    uniq = {}
    for d in out:
        uniq[d["name"]] = d
    return list(uniq.values())


@lru_cache(maxsize=32)
def fetch_ephemerides_cached(target: str) -> tuple[dict, ...]:
    """LRU cached wrapper used by the app; returns tuple for hashability."""
    out = fetch_ephemerides(target)
    return tuple(out)


def _generic_table_fetch(url: str, cache_path: str | Path, force: bool, timeout: int = 60) -> pd.DataFrame:
    cache = Path(cache_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and not force:
        try:
            return pd.read_csv(cache)
        except Exception:
            pass
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    try:
        df.to_csv(cache, index=False)
    except Exception:
        pass
    return df


def fetch_toi_table(cache_path: str | Path = "data/toi_cache.csv", force: bool = False) -> pd.DataFrame:
    return _generic_table_fetch(TOI_URL, cache_path, force)


def fetch_k2_table(cache_path: str | Path = "data/k2_cache.csv", force: bool = False) -> pd.DataFrame:
    return _generic_table_fetch(K2_URL, cache_path, force)


def fetch_light_curve(target: str, mission: str = "Kepler", timeout: int = 60):
    try:
        if mission.lower() == "kepler":
            search = lk.search_lightcurve(target, mission="Kepler")
        elif mission.lower() == "k2":
            search = lk.search_lightcurve(target, mission="K2")
        elif mission.lower() == "tess":
            search = lk.search_lightcurve(target, mission="TESS")
        else:
            raise ValueError("Unknown mission: " + mission)
        lc_collection = search.download_all(timeout=timeout)
        if lc_collection is None or len(lc_collection) == 0:
            raise RuntimeError("No light curve found for target.")
        lc = lc_collection.stitch()
        return lc.time.value, lc.flux.value
    except Exception as e:
        print(f"Data fetch failed for {target} ({mission}): {e}")
        return None, None


if __name__ == "__main__":  # Manual quick smoke (avoid network on import during tests)
    t, f = fetch_light_curve("Kepler-10", mission="Kepler")
    if t is not None:
        print("Fetched points:", len(t))



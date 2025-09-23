import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# Convenience alias for tooling
FILE = __file__
file = __file__

EXOPLANET_API_BASE = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _retry_session(total=5, backoff=0.5) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=total,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    ad = HTTPAdapter(max_retries=retry)
    s.mount("https://", ad)
    s.mount("http://", ad)
    return s


_SESSION = _retry_session()


def _api_url(table: str) -> str:
    return f"{EXOPLANET_API_BASE}?table={table}&select=*&format=csv"


FEATURE_COLS = [
    "koi_period",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_steff",
    "koi_slogg",
    "koi_srad",
]
LABEL_COL = "koi_disposition"


def fetch_koi_table(
    cache_path: str | Path = "data/koi_cache.csv", force: bool = False
) -> pd.DataFrame:
    cache_path = Path(cache_path)
    ensure_dir(cache_path.parent)
    if cache_path.exists() and not force:
        return pd.read_csv(cache_path)
    # Important: use requests.get so tests can monkeypatch nasa.requests.get
    r = requests.get(_api_url("q1_q17_dr25_koi"), timeout=60)
    try:
        r.raise_for_status()
    except Exception:
        pass
    df = pd.read_csv(io.StringIO(r.text))
    df.to_csv(cache_path, index=False)
    return df


def fetch_toi_table(
    cache_path: str | Path = "data/toi_cache.csv", force: bool = False
) -> pd.DataFrame:
    cache_path = Path(cache_path)
    ensure_dir(cache_path.parent)
    if cache_path.exists() and not force:
        return pd.read_csv(cache_path)
    r = requests.get(_api_url("toi"), timeout=60)
    try:
        r.raise_for_status()
    except Exception:
        pass
    df = pd.read_csv(io.StringIO(r.text))
    df.to_csv(cache_path, index=False)
    return df


def fetch_k2_table(
    cache_path: str | Path = "data/k2_cache.csv", force: bool = False
) -> pd.DataFrame:
    cache_path = Path(cache_path)
    ensure_dir(cache_path.parent)
    if cache_path.exists() and not force:
        return pd.read_csv(cache_path)
    r = requests.get(_api_url("k2candidates"), timeout=60)
    try:
        r.raise_for_status()
    except Exception:
        pass
    df = pd.read_csv(io.StringIO(r.text))
    if any("ERROR" in str(c).upper() for c in df.columns):
        logger.warning("K2 API returned an error page; continuing with empty K2 table")
        df = pd.DataFrame(
            columns=[
                "epic",
                "period",
                "duration",
                "depth",
                "planet_radius",
                "st_teff",
                "st_logg",
                "st_rad",
                "disposition",
            ]
        ).iloc[0:0]
    df.to_csv(cache_path, index=False)
    return df


def clean_koi(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(dict.fromkeys(FEATURE_COLS + [LABEL_COL, "kepid", "kepoi_name", "kepler_name"]))
    cols = [c for c in cols if c in df.columns]
    out = df.loc[:, cols].copy()
    out[LABEL_COL] = out[LABEL_COL].astype(str).str.upper().str.strip()
    return out


def _cf(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.casefold()


def koi_row_for_target(df: pd.DataFrame, target: str) -> pd.Series | None:
    if not len(df):
        return None
    name = str(target).strip()
    mask = pd.Series(False, index=df.index)
    if "kepler_name" in df.columns:
        mask = mask | (_cf(df["kepler_name"]) == name.casefold())
    if "kepoi_name" in df.columns:
        mask = mask | (_cf(df["kepoi_name"]) == name.casefold())
    if "kepid" in df.columns and name.isdigit():
        mask = mask | (df["kepid"].astype(str) == name)
    hits = df[mask]
    if hits.empty:
        return None
    return hits.iloc[0]


def _std_features_from_row(row: pd.Series, mapping: dict) -> np.ndarray:
    vals = [
        pd.to_numeric(row.get(mapping[k], np.nan), errors="coerce")
        for k in ["period", "duration", "depth", "prad", "teff", "logg", "srad"]
    ]
    # Normalize depth if it looks like ppm
    if np.isfinite(vals[2]) and vals[2] and vals[2] > 1:
        vals[2] = vals[2] / 1e6
    return np.array(vals, dtype=float)


def resolve_features_for_target(target: str) -> dict | None:
    t = str(target).strip()
    # KOI
    try:
        dfk = clean_koi(fetch_koi_table())
        r = koi_row_for_target(dfk, t)
        if r is not None:
            mp = {
                "period": "koi_period",
                "duration": "koi_duration",
                "depth": "koi_depth",
                "prad": "koi_prad",
                "teff": "koi_steff",
                "logg": "koi_slogg",
                "srad": "koi_srad",
            }
            feats = _std_features_from_row(r, mp)
            return {"source": "KOI", "row": r, "features": feats}
    except Exception:
        pass
    # TOI
    try:
        dft = fetch_toi_table()
        name_mask = _cf(dft.get("toi", pd.Series([], dtype=str))).eq(t.casefold()) | _cf(
            dft.get("tic_id", pd.Series([], dtype=str))
        ).eq(t.casefold())
        if name_mask.any():
            row = dft[name_mask].iloc[0]
            mp = {
                "period": "orbital_period",
                "duration": "transit_duration",
                "depth": "transit_depth",
                "prad": "pl_rade",
                "teff": "st_teff",
                "logg": "st_logg",
                "srad": "st_rad",
            }
            feats = _std_features_from_row(row, mp)
            return {"source": "TOI", "row": row, "features": feats}
    except Exception:
        pass
    # K2
    try:
        df2 = fetch_k2_table()
        epic = t.replace("EPIC", "").strip()
        name_mask = _cf(df2.get("epic", pd.Series([], dtype=str))).eq(epic)
        if name_mask.any():
            row = df2[name_mask].iloc[0]
            mp = {
                "period": "period",
                "duration": "duration",
                "depth": "depth",
                "prad": "planet_radius",
                "teff": "st_teff",
                "logg": "st_logg",
                "srad": "st_rad",
            }
            feats = _std_features_from_row(row, mp)
            return {"source": "K2", "row": row, "features": feats}
    except Exception:
        pass
    return None

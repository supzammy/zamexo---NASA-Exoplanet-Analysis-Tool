import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from astropy.timeseries import BoxLeastSquares
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

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


def norm_name(s: str) -> str:
    return str(s).lower().replace(" ", "").replace("-", "")


def koi_row_for_target(df: pd.DataFrame, target: str) -> pd.Series | None:
    if df is None or len(df) == 0:
        return None
    key = norm_name(target)
    for col in ("kepler_name", "kepoi_name"):
        if col in df.columns:
            m = df[col].astype(str).map(norm_name) == key
            if m.any():
                return df.loc[m].iloc[0]
    return None


def clean_koi(df: pd.DataFrame) -> pd.DataFrame:
    cols = FEATURE_COLS + [LABEL_COL, "kepid", "kepoi_name", "kepler_name"]
    keep = [c for c in cols if c in df.columns]
    return df[keep].copy()


def _clean_series(time: np.ndarray, flux: np.ndarray):
    t = np.asarray(time, dtype=float)
    f = np.asarray(flux, dtype=float)
    m = np.isfinite(t) & np.isfinite(f)
    t, f = t[m], f[m]
    if t.size:
        med = np.nanmedian(f)
        if np.isfinite(med) and med != 0:
            f = f / med
    return t, f


def fold_curve(time: np.ndarray, period: float, t0: float):
    t = np.asarray(time, dtype=float)
    phase = ((t - t0 + 0.5 * period) % period) / period - 0.5
    order = np.argsort(phase)
    return phase[order], order


def bls_features(
    time: np.ndarray,
    flux: np.ndarray,
    max_period: float = 30.0,
    min_period: float = 0.2,
    dur_fracs: tuple[float, float, int] = (0.01, 0.1, 200),
):
    t, f = _clean_series(time, flux)
    if t.size < 100:
        return {
            "period": np.nan,
            "duration": np.nan,
            "depth": np.nan,
            "t0": np.nan,
            "power": np.nan,
            "sde": np.nan,
            "n_points": int(t.size),
            "time_span": float((t.max() - t.min()) if t.size else 0.0),
        }
    y = f - 1.0
    durations = np.linspace(dur_fracs[0], dur_fracs[1], int(dur_fracs[2])) * (
        max_period if np.isfinite(max_period) else (t.max() - t.min())
    )
    bls = BoxLeastSquares(t, y)
    res = bls.autopower(
        durations, minimum_period=min_period, maximum_period=max_period, objective="snr"
    )
    k = int(np.nanargmax(res.power))
    period = float(res.period[k])
    duration = float(res.duration[k])
    t0 = float(res.transit_time[k])
    power = float(res.power[k])
    mu, sig = float(np.nanmean(res.power)), float(np.nanstd(res.power))
    sde = (power - mu) / (sig if sig > 0 else 1.0)
    model = bls.model(t, period, duration, t0)
    depth = float(abs(np.nanmedian(1.0 + model) - 1.0))
    return {
        "period": period,
        "duration": duration,
        "depth": depth,
        "t0": t0,
        "power": power,
        "sde": float(sde),
        "n_points": int(t.size),
        "time_span": float(t.max() - t.min()),
    }


EXOPLANET_API_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"
    "?table=q1_q17_dr25_koi&select=*&format=csv"
)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def fetch_koi_table(
    cache_path: str | Path = "data/koi_cache.csv", force: bool = False
) -> pd.DataFrame:
    cache_path = Path(cache_path)
    ensure_dir(cache_path.parent)
    if cache_path.exists() and not force:
        return pd.read_csv(cache_path)
    r = requests.get(EXOPLANET_API_URL, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.to_csv(cache_path, index=False)
    return df


OUT_DIR = Path("models")
LABEL_MAP = {"FALSE POSITIVE": 0, "CANDIDATE": 1, "CONFIRMED": 2}


def main(random_state: int = 42) -> float:
    print("📥 Loading KOI…")
    df_raw = fetch_koi_table()
    df = clean_koi(df_raw)
    y = df[LABEL_COL].map(LABEL_MAP)
    X = df[FEATURE_COLS].copy()

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    print("🧠 Training RandomForest…")
    clf = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, random_state=random_state, n_jobs=-1
    )
    clf.fit(X_tr, y_tr)

    print("📊 Evaluating…")
    y_pr = clf.predict(X_va)
    acc = accuracy_score(y_va, y_pr)
    report = classification_report(
        y_va, y_pr, target_names=list(LABEL_MAP.keys()), output_dict=True
    )
    print(f"✅ Accuracy: {acc:.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, OUT_DIR / "model.joblib")
    (OUT_DIR / "feature_cols.json").write_text(json.dumps(FEATURE_COLS, indent=2))
    (OUT_DIR / "metrics.json").write_text(json.dumps({"accuracy": acc, "report": report}, indent=2))
    print(f"💾 Saved to {OUT_DIR.resolve()}")
    return acc


if __name__ == "__main__":
    main()

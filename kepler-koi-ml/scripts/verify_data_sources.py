import hashlib
import json
from pathlib import Path

import pandas as pd

from scripts.train_multisource import CANDIDATES, pick_col  # reuse mapping
from utils.nasa import (
    FEATURE_COLS as KOI_FEATURES,
    LABEL_COL as KOI_LABEL,
    fetch_k2_table,
    fetch_koi_table,
    fetch_toi_table,
)

DATA_DIR = Path("data")
CACHE_FILES = {
    "KOI": DATA_DIR / "koi_cache.csv",
    "TOI": DATA_DIR / "toi_cache.csv",
    "K2": DATA_DIR / "k2_cache.csv",
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def df_summary(df: pd.DataFrame, name: str) -> dict:
    out = {
        "name": name,
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "sample_cols": list(df.columns[:12]),
    }
    return out


def verify_koi() -> dict:
    df = fetch_koi_table()
    ok_cols = all(c in df.columns for c in KOI_FEATURES + [KOI_LABEL])
    return {"summary": df_summary(df, "KOI"), "has_expected_cols": bool(ok_cols)}


def verify_generic(fetch_fn, label_candidates: list[str], name: str) -> dict:
    df = fetch_fn()
    lbl = pick_col(df, label_candidates)
    return {"summary": df_summary(df, name), "label_col_detected": lbl}


def main():
    results = {}
    results["KOI"] = verify_koi()
    results["TOI"] = verify_generic(fetch_toi_table, CANDIDATES["label"], "TOI")
    results["K2"] = verify_generic(fetch_k2_table, CANDIDATES["label"], "K2")

    for k, p in CACHE_FILES.items():
        if p.exists():
            results[k]["cache"] = {
                "path": str(p.resolve()),
                "size_bytes": p.stat().st_size,
                "sha256": sha256_file(p),
                "mtime_iso": pd.Timestamp(p.stat().st_mtime, unit="s").isoformat(),
            }
        else:
            results[k]["cache"] = {"missing": True}

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

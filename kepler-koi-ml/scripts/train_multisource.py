import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from utils.nasa import (
    LABEL_COL as KOI_LABEL,
    clean_koi,
    fetch_k2_table,
    fetch_koi_table,
    fetch_toi_table,
)

OUT_DIR = Path("models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATES = {
    "period": ["koi_period", "orbital_period", "period", "pl_orbper"],
    "duration": ["koi_duration", "transit_duration", "duration", "dur", "tran_dur"],
    "depth": ["koi_depth", "transit_depth", "depth", "depth_ppm"],
    "prad": ["koi_prad", "pl_rade", "planet_radius", "prad"],
    "teff": ["koi_steff", "st_teff", "teff"],
    "logg": ["koi_slogg", "st_logg", "logg"],
    "srad": ["koi_srad", "st_rad", "srad"],
    "name": ["kepler_name", "kepoi_name", "toi", "tic_id", "tic", "epic", "kepid"],
    "label": ["koi_disposition", "tfopwg_disp", "k2koi_disposition", "disposition"],
}

LABEL_MAP = {"FALSE POSITIVE": 0, "CANDIDATE": 1, "CONFIRMED": 2}


def pick_col(df: pd.DataFrame, keys: list[str]) -> str | None:
    for k in keys:
        if k in df.columns:
            return k
    return None


def normalize_depth(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    med = np.nanmedian(s)
    if np.isfinite(med) and med > 1:
        return s / 1e6  # ppm -> fraction
    return s


def to_standard_frame(df: pd.DataFrame, source: str) -> pd.DataFrame:
    out = pd.DataFrame()
    for key in ["period", "duration", "depth", "prad", "teff", "logg", "srad"]:
        col = pick_col(df, CANDIDATES[key])
        out[key] = pd.to_numeric(df[col], errors="coerce") if col else np.nan
    if out["depth"].notna().any():
        out["depth"] = normalize_depth(out["depth"])
    lbl_col = pick_col(df, CANDIDATES["label"])
    y = (
        df[lbl_col].astype("string").str.upper().str.strip()
        if lbl_col
        else pd.Series(index=df.index, dtype="string")
    )
    if source == "TOI":
        y = y.replace(
            {"PC": "CANDIDATE", "FP": "FALSE POSITIVE", "KP": "CONFIRMED", "APC": "CANDIDATE"}
        ).infer_objects(copy=False)
    elif source == "K2":
        y = y.replace({"FP": "FALSE POSITIVE", "PC": "CANDIDATE"}).infer_objects(copy=False)
    out["label"] = y
    out["source"] = source
    name_col = pick_col(df, CANDIDATES["name"])
    out["name"] = df[name_col].astype(str) if name_col else ""
    return out


def _save_confusion_matrix(cm, labels, out_path: Path):
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main(sources: list[str], random_state: int = 42):
    print(f"Sources: {sources}")
    frames = []

    if "KOI" in sources or "ALL" in sources:
        print("📥 Fetching KOI...")
        dfk = clean_koi(fetch_koi_table())
        print(f"KOI rows: {len(dfk)}")
        fk = pd.DataFrame()
        # Map KOI columns into standard features
        mapping = {
            "period": "koi_period",
            "duration": "koi_duration",
            "depth": "koi_depth",
            "prad": "koi_prad",
            "teff": "koi_steff",
            "logg": "koi_slogg",
            "srad": "koi_srad",
        }
        for key, col in mapping.items():
            fk[key] = pd.to_numeric(dfk[col], errors="coerce")
        fk["depth"] = normalize_depth(fk["depth"])
        fk["label"] = dfk[KOI_LABEL].astype(str).str.upper().str.strip()
        fk["source"] = "KOI"
        fk["name"] = dfk.get("kepler_name", dfk.get("kepoi_name", ""))
        frames.append(fk)

    if "TOI" in sources or "ALL" in sources:
        print("📥 Fetching TOI...")
        dft = fetch_toi_table()
        print(f"TOI rows: {len(dft)}")
        ft = to_standard_frame(dft, "TOI")
        frames.append(ft)

    if "K2" in sources or "ALL" in sources:
        print("📥 Fetching K2...")
        df2 = fetch_k2_table()
        print(f"K2 rows: {len(df2)}")
        f2 = to_standard_frame(df2, "K2")
        frames.append(f2)

    if not frames:
        raise SystemExit("No sources selected")

    df = pd.concat(frames, ignore_index=True)
    print(f"Combined rows: {len(df)}")

    X = df[["period", "duration", "depth", "prad", "teff", "logg", "srad"]]
    y = df["label"].map(LABEL_MAP)
    mask = y.notna()
    X, y = X[mask], y[mask]

    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=400, min_samples_leaf=2, random_state=random_state, n_jobs=-1
                ),
            ),
        ]
    )

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    print("🧠 Training RandomForest…")
    pipe.fit(X_tr, y_tr)
    y_pr = pipe.predict(X_va)
    acc = accuracy_score(y_va, y_pr)
    report = classification_report(
        y_va, y_pr, target_names=list(LABEL_MAP.keys()), output_dict=True
    )
    cm = confusion_matrix(y_va, y_pr, labels=[0, 1, 2])
    print(f"✅ Multi-source accuracy: {acc:.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, OUT_DIR / "rf_multi.joblib")
    (OUT_DIR / "multi_feature_cols.json").write_text(json.dumps(list(X.columns), indent=2))
    (OUT_DIR / "multi_metrics.json").write_text(
        json.dumps(
            {
                "accuracy": acc,
                "sources": sources,
                "report": report,
                "labels": ["FALSE POSITIVE", "CANDIDATE", "CONFIRMED"],
                "confusion_matrix": cm.tolist(),
            },
            indent=2,
        )
    )
    _save_confusion_matrix(
        cm, ["FALSE POSITIVE", "CANDIDATE", "CONFIRMED"], OUT_DIR / "multi_confusion_matrix.png"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="KOI", help="Comma-separated: KOI,TOI,K2,ALL")
    args = ap.parse_args()
    sources = [s.strip().upper() for s in args.sources.split(",")]
    main(sources)

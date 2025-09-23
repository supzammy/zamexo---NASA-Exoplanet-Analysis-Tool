import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from utils.nasa import FEATURE_COLS, LABEL_COL, clean_koi, fetch_koi_table

OUT_DIR = Path("models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {"FALSE POSITIVE": 0, "CANDIDATE": 1, "CONFIRMED": 2}


def _save_cm(cm, labels, out):
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
    fig.savefig(out, dpi=160)
    plt.close(fig)


def train(df: pd.DataFrame, random_state: int = 42):
    X = df[FEATURE_COLS].copy()
    X = X.apply(pd.to_numeric, errors="coerce").astype(float)
    # Normalize depth from ppm to fraction if needed
    if "koi_depth" in X.columns:
        mask = X["koi_depth"] > 1
        X.loc[mask, "koi_depth"] = X.loc[mask, "koi_depth"] / 1e6
    y = df[LABEL_COL].map(LABEL_MAP)
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

    # Only stratify if each class has >= 2 samples
    vc = pd.Series(y).value_counts()
    strat = y if (len(vc) > 1 and vc.min() >= 2) else None

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=strat
    )
    pipe.fit(X_tr, y_tr)
    y_pr = pipe.predict(X_va)

    labels = [0, 1, 2]
    acc = float(accuracy_score(y_va, y_pr)) if len(y_va) else float("nan")
    report = classification_report(
        y_va,
        y_pr,
        labels=labels,
        target_names=list(LABEL_MAP.keys()),
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_va, y_pr, labels=labels)
    return pipe, acc, report, cm, list(X.columns)


def train_main(out_dir: Path = OUT_DIR) -> float:
    df = clean_koi(fetch_koi_table())
    pipe, acc, report, cm, cols = train(df)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_dir / "model.joblib")
    (out_dir / "feature_cols.json").write_text(json.dumps(cols, indent=2))
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "accuracy": acc,
                "labels": ["FALSE POSITIVE", "CANDIDATE", "CONFIRMED"],
                "report": report,
                "confusion_matrix": cm.tolist(),
            },
            indent=2,
        )
    )
    # Also keep KOI-prefixed artifacts
    joblib.dump(pipe, out_dir / "rf_koi.joblib")
    (out_dir / "koi_feature_cols.json").write_text(json.dumps(cols, indent=2))
    (out_dir / "koi_metrics.json").write_text(
        json.dumps(
            {
                "accuracy": acc,
                "labels": ["FALSE POSITIVE", "CANDIDATE", "CONFIRMED"],
                "report": report,
                "confusion_matrix": cm.tolist(),
            },
            indent=2,
        )
    )
    _save_cm(cm, ["FALSE POSITIVE", "CANDIDATE", "CONFIRMED"], out_dir / "koi_confusion_matrix.png")
    print(f"✅ Baseline KOI accuracy: {acc:.3f}")
    return float(acc)


def main():
    return train_main()


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()

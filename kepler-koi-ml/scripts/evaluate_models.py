import json
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from utils.nasa import FEATURE_COLS, LABEL_COL, clean_koi, fetch_koi_table

MODELS = Path("models")


def eval_baseline(random_state=42):
    df = clean_koi(fetch_koi_table())
    X, y = df[FEATURE_COLS].copy(), df[LABEL_COL].map(
        {"FALSE POSITIVE": 0, "CANDIDATE": 1, "CONFIRMED": 2}
    )
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    clf = joblib.load(MODELS / "model.joblib")
    acc = accuracy_score(y_va, clf.predict(X_va))
    saved = json.loads((MODELS / "metrics.json").read_text())
    return {"recomputed_acc": float(acc), "saved_acc": float(saved.get("accuracy", float("nan")))}


def main():
    out = {"baseline": eval_baseline()}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

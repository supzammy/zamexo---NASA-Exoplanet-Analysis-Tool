from __future__ import annotations
import json
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from utils.nasa import fetch_koi_table, prepare_koi_training


def main() -> float:
    models_dir = Path("models"); models_dir.mkdir(parents=True, exist_ok=True)
    df_all = fetch_koi_table("data/koi_cache.csv", force=False)
    df = prepare_koi_training(df_all)
    y = df["koi_disposition"].astype(str).to_numpy()
    X = df.drop(columns=["koi_disposition"])
    feature_cols = list(X.columns)
    Xtr, Xva, ytr, yva = train_test_split(X.to_numpy(), y, test_size=0.2, random_state=42, stratify=y)
    clf = joblib.Parallel if False else None  # no-op; keeps file short
    model = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1, class_weight="balanced_subsample").fit(Xtr, ytr)
    yhat = model.predict(Xva)
    acc = float(accuracy_score(yva, yhat))
    joblib.dump(model, models_dir / "model.joblib")
    (models_dir / "feature_cols.json").write_text(json.dumps(feature_cols, indent=2))
    (models_dir / "metrics.json").write_text(json.dumps({"accuracy": acc, "report": classification_report(yva, yhat, output_dict=True)}, indent=2))
    print(f"Validation accuracy: {acc:.3f}")
    return acc


if __name__ == "__main__":
    main()

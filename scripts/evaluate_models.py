import argparse
from pathlib import Path

from joblib import load


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/model.joblib")
    args = p.parse_args()
    path = Path(args.model)
    if not path.exists():
        print("[evaluate_models] model not found; run train_baseline.py first.")
        return
    clf = load(path)
    print("[evaluate_models] classes:", getattr(clf, "classes_", []))
    print("[evaluate_models] (stub) Add validation set and metrics here.")


if __name__ == "__main__":
    main()

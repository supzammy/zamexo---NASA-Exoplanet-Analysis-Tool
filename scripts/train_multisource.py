import argparse
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=1)
    args = p.parse_args()
    for e in range(args.epochs):
        time.sleep(0.05)
        print(f"[train_multisource] epoch {e + 1}/{args.epochs} done")
    print("[train_multisource] completed (stub)")


if __name__ == "__main__":
    main()

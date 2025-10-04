import numpy as np

from utils.features import run_bls


def main():
    # Synthetic transit
    t = np.linspace(0, 10, 2000)
    f = np.ones_like(t)
    f[((t % 2.5) < 0.1)] -= 0.005
    bls = run_bls(t, f, max_period=5.0)
    print("[verify_bls]", bls)


if __name__ == "__main__":
    main()

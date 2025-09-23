import argparse

import numpy as np
from lightkurve import search_lightcurve

from utils.features import bls_features


def fetch_lc(target: str):
    for mission, author in [("Kepler", None), ("K2", None), ("TESS", "SPOC")]:
        sr = search_lightcurve(target, mission=mission, author=author, cadence="long")
        if len(sr) == 0:
            continue
        lcs = sr.download_all()
        return lcs.stitch().remove_nans().flatten(window_length=401)
    return None


def main(target: str, expect_period: float | None, tol: float):
    lc = fetch_lc(target)
    if lc is None:
        print("No LC found.")
        return 2
    t, f = lc.time.value, (lc.flux.value / np.nanmedian(lc.flux.value))
    feats = bls_features(t, f, max_period=50.0)
    print(
        "Target="
        f"{target} "
        f"period={feats['period']:.5f} d "
        f"sde={feats['sde']:.2f} "
        f"depth={feats['depth']:.6f}"
    )
    if expect_period is not None and np.isfinite(feats["period"]):
        if abs(feats["period"] - expect_period) <= tol:
            print("OK: period within tolerance")
            return 0
        else:
            print("WARN: period out of tolerance")
            return 1
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Kepler-10")
    ap.add_argument("--expect-period", type=float, default=None, help="Expected period in days")
    ap.add_argument("--tol", type=float, default=0.1, help="Tolerance in days")
    args = ap.parse_args()
    raise SystemExit(main(args.target, args.expect_period, args.tol))

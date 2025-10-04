from __future__ import annotations

import sys

from utils.nasa import get_lightcurve_cached


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python -m scripts.prefetch_lc <mission: Kepler|K2|TESS> <target1> [target2 ...]"
        )
        sys.exit(2)
    mission = sys.argv[1]
    for target in sys.argv[2:]:
        print(f"Prefetching {mission} {target} …")
        t, f = get_lightcurve_cached(target, mission, refresh=True, max_rows=60000)
        print("OK" if t is not None else "Not found")


if __name__ == "__main__":
    main()

from pathlib import Path

from utils.nasa import fetch_koi_table


def main():
    path = fetch_koi_table(Path("data/koi_cache.csv"))
    print("[verify_data_sources] KOI cache at:", path)


if __name__ == "__main__":
    main()

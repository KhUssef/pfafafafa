import argparse
import csv
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Iterate over world cities and print the number of sub cities "
            "(cities in the same country+subcountry group)."
        )
    )
    parser.add_argument(
        "--input-path",
        default="world-cities.csv",
        help="Path to world-cities CSV file.",
    )
    parser.add_argument(
        "--city-column",
        default="name",
        help="City column name.",
    )
    parser.add_argument(
        "--country-column",
        default="country",
        help="Country column name.",
    )
    parser.add_argument(
        "--subcountry-column",
        default="subcountry",
        help="Subcountry column name.",
    )
    parser.add_argument(
        "--exclude-self",
        action="store_true",
        help="If set, do not count the city itself in sub city count.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit for printing rows (useful for quick checks).",
    )
    return parser.parse_args()


def normalize(value):
    return (value or "").strip().casefold()


def load_rows(csv_path):
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header row: {csv_path}")
        return reader.fieldnames, list(reader)


def main():
    args = parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    fieldnames, rows = load_rows(input_path)

    required = [args.city_column, args.country_column, args.subcountry_column]
    missing = [col for col in required if col not in fieldnames]
    if missing:
        available = ", ".join(fieldnames)
        raise ValueError(
            f"Missing columns: {', '.join(missing)}. Available columns: {available}"
        )

    group_counts = Counter()
    for row in rows:
        key = (
            normalize(row.get(args.country_column, "")),
            normalize(row.get(args.subcountry_column, "")),
        )
        group_counts[key] += 1

    printed = 0
    print("city,country,subcountry,subcity_count")
    for row in rows:
        key = (
            normalize(row.get(args.country_column, "")),
            normalize(row.get(args.subcountry_column, "")),
        )
        count = group_counts.get(key, 0)
        if args.exclude_self and count > 0:
            count -= 1

        city = (row.get(args.city_column) or "").strip()
        country = (row.get(args.country_column) or "").strip()
        subcountry = (row.get(args.subcountry_column) or "").strip()
        print(f"{city},{country},{subcountry},{count}")

        printed += 1
        if args.max_rows is not None and printed >= args.max_rows:
            break


if __name__ == "__main__":
    main()

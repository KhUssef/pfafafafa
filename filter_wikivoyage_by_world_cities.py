import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Filter wikivoyage-listings-en.csv to keep only rows whose article matches "
            "city names from world-cities-with-occurrences.csv."
        )
    )
    parser.add_argument(
        "--cities-occurrences-path",
        default="new/world-cities-with-occurrences.csv",
        help="Path to world cities CSV with article_occurrences column.",
    )
    parser.add_argument(
        "--cities-name-column",
        default="city",
        help="Column name in the cities CSV that contains city names.",
    )
    parser.add_argument(
        "--occurrences-column",
        default="article_occurrences",
        help="Column name in the cities CSV that contains occurrence counts.",
    )
    parser.add_argument(
        "--min-occurrences",
        type=int,
        default=1,
        help="Only include city names with occurrences >= this threshold.",
    )
    parser.add_argument(
        "--listings-path",
        default="wikivoyage-listings-en.csv",
        help="Path to input listings CSV.",
    )
    parser.add_argument(
        "--listings-article-column",
        default="article",
        help="Article column name in listings CSV.",
    )
    parser.add_argument(
        "--output-path",
        default="new/wikivoyage-listings-en-matching-cities.csv",
        help="Path to filtered listings output CSV.",
    )
    return parser.parse_args()


def normalize_name(value):
    if value is None:
        return ""
    return str(value).strip().casefold()


def load_allowed_city_names(
    cities_occurrences_path,
    cities_name_column,
    occurrences_column,
    min_occurrences,
):
    allowed = set()
    with open(cities_occurrences_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header row: {cities_occurrences_path}")
        if cities_name_column not in reader.fieldnames:
            available = ", ".join(reader.fieldnames)
            raise ValueError(
                f"Missing city name column '{cities_name_column}'. Available: {available}"
            )
        if occurrences_column not in reader.fieldnames:
            available = ", ".join(reader.fieldnames)
            raise ValueError(
                f"Missing occurrences column '{occurrences_column}'. Available: {available}"
            )

        for row in reader:
            raw_name = row.get(cities_name_column, "")
            raw_occurrences = row.get(occurrences_column, "0")
            try:
                occurrences = int(raw_occurrences)
            except ValueError:
                occurrences = 0

            if occurrences >= min_occurrences:
                normalized = normalize_name(raw_name)
                if normalized:
                    allowed.add(normalized)

    return allowed


def filter_listings(listings_path, listings_article_column, allowed_city_names, output_path):
    with open(listings_path, "r", encoding="utf-8", newline="") as in_handle:
        reader = csv.DictReader(in_handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header row: {listings_path}")
        if listings_article_column not in reader.fieldnames:
            available = ", ".join(reader.fieldnames)
            raise ValueError(
                f"Missing listings article column '{listings_article_column}'. Available: {available}"
            )

        total_rows = 0
        kept_rows = 0
        with open(output_path, "w", encoding="utf-8", newline="") as out_handle:
            writer = csv.DictWriter(out_handle, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                total_rows += 1
                article_name = row.get(listings_article_column, "")
                normalized_article = normalize_name(article_name)
                if normalized_article in allowed_city_names:
                    writer.writerow(row)
                    kept_rows += 1

    return total_rows, kept_rows


def main():
    args = parse_args()

    cities_occurrences_path = Path(args.cities_occurrences_path)
    listings_path = Path(args.listings_path)
    output_path = Path(args.output_path)

    if not cities_occurrences_path.exists():
        raise FileNotFoundError(
            f"Cities occurrences CSV not found: {cities_occurrences_path}"
        )
    if not listings_path.exists():
        raise FileNotFoundError(f"Listings CSV not found: {listings_path}")

    allowed_city_names = load_allowed_city_names(
        cities_occurrences_path=cities_occurrences_path,
        cities_name_column=args.cities_name_column,
        occurrences_column=args.occurrences_column,
        min_occurrences=args.min_occurrences,
    )

    total_rows, kept_rows = filter_listings(
        listings_path=listings_path,
        listings_article_column=args.listings_article_column,
        allowed_city_names=allowed_city_names,
        output_path=output_path,
    )

    print(f"Allowed city names loaded: {len(allowed_city_names)}")
    print(f"Listings rows scanned: {total_rows}")
    print(f"Listings rows kept: {kept_rows}")
    print(f"Filtered output saved to: {output_path}")


if __name__ == "__main__":
    main()

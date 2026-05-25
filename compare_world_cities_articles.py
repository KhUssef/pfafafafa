import argparse
import csv
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Count article occurrences from metadata and map them to world cities, "
            "including cities with zero matches."
        )
    )
    parser.add_argument(
        "--metadata-path",
        default="wikivoyage-listings-en.csv",
        help="Path to metadata CSV containing article names.",
    )
    parser.add_argument(
        "--cities-path",
        default="new\\merged-world-wikivoyage-descriptions.csv",
        help="Path to world cities CSV.",
    )
    parser.add_argument(
        "--city-column",
        default="city",
        help="City name column in world cities CSV.",
    )
    parser.add_argument(
        "--article-key",
        default="article",
        help="Metadata field that contains article names.",
    )
    parser.add_argument(
        "--output-path",
        default="new/world-cities-with-occurrences.csv",
        help="Output CSV path with city occurrence counts.",
    )
    parser.add_argument(
        "--article-counts-output",
        default="new/article-counts.csv",
        help="Output CSV path for unique article occurrence counts.",
    )
    return parser.parse_args()


def normalize_name(value):
    if value is None:
        return ""
    return str(value).strip().casefold()


def load_article_counts(metadata_path, article_key):
    with open(metadata_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header row: {metadata_path}")
        if article_key not in reader.fieldnames:
            available = ", ".join(reader.fieldnames)
            raise ValueError(
                f"Article column '{article_key}' not found in {metadata_path}. Available columns: {available}"
            )

        counts = Counter()
        original_name_by_normalized = {}
        for item in reader:
            article_name = item.get(article_key, "")
            normalized = normalize_name(article_name)
            if not normalized:
                continue
            counts[normalized] += 1
            original_name_by_normalized.setdefault(normalized, str(article_name).strip())

    return counts, original_name_by_normalized


def write_article_counts_csv(output_path, counts, original_name_by_normalized):
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["article", "occurrences"])
        for normalized_name, occurrence_count in counts.most_common():
            writer.writerow(
                [
                    original_name_by_normalized.get(normalized_name, normalized_name),
                    int(occurrence_count),
                ]
            )


def map_counts_to_world_cities(cities_path, city_column, article_counts):
    with open(cities_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header row: {cities_path}")
        if city_column not in reader.fieldnames:
            available = ", ".join(reader.fieldnames)
            raise ValueError(
                f"City column '{city_column}' not found in {cities_path}. Available columns: {available}"
            )

        rows = []
        for row in reader:
            city_name = row.get(city_column, "")
            normalized_city = normalize_name(city_name)
            row["article_occurrences"] = int(article_counts.get(normalized_city, 0))
            rows.append(row)

        fieldnames = list(reader.fieldnames) + ["article_occurrences"]

    return rows, fieldnames


def write_city_output_csv(output_path, rows, fieldnames):
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()

    metadata_path = Path(args.metadata_path)
    cities_path = Path(args.cities_path)
    output_path = Path(args.output_path)
    article_counts_output = Path(args.article_counts_output)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")
    if not cities_path.exists():
        raise FileNotFoundError(f"World cities CSV not found: {cities_path}")

    article_counts, original_name_by_normalized = load_article_counts(
        metadata_path=metadata_path,
        article_key=args.article_key,
    )

    write_article_counts_csv(
        output_path=article_counts_output,
        counts=article_counts,
        original_name_by_normalized=original_name_by_normalized,
    )

    rows, fieldnames = map_counts_to_world_cities(
        cities_path=cities_path,
        city_column=args.city_column,
        article_counts=article_counts,
    )

    write_city_output_csv(
        output_path=output_path,
        rows=rows,
        fieldnames=fieldnames,
    )

    matched_city_rows = sum(1 for row in rows if int(row["article_occurrences"]) > 0)
    print(f"Unique articles counted: {len(article_counts)}")
    print(f"World city rows processed: {len(rows)}")
    print(f"World city rows with matches: {matched_city_rows}")
    print(f"Saved article counts to: {article_counts_output}")
    print(f"Saved world city occurrences to: {output_path}")


if __name__ == "__main__":
    main()

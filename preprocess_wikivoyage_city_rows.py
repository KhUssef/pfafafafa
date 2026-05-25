import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess matched Wikivoyage listings and keep only city, country, "
            "type, title, and description."
        )
    )
    parser.add_argument(
        "--input-path",
        default="new/wikivoyage-listings-en-matching-cities.csv",
        help="Input CSV produced by filter_wikivoyage_by_world_cities.py",
    )
    parser.add_argument(
        "--world-cities-path",
        default="new/world-cities-with-occurrences.csv",
        help="World cities CSV used to map city names to country.",
    )
    parser.add_argument(
        "--city-column-in-input",
        default="article",
        help="Column in input CSV that contains city/article name.",
    )
    parser.add_argument(
        "--city-column-in-world",
        default="city",
        help="City name column in world cities CSV.",
    )
    parser.add_argument(
        "--country-column-in-world",
        default="country",
        help="Country column in world cities CSV.",
    )
    parser.add_argument(
        "--output-path",
        default="new/wikivoyage-city-preprocessed.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def normalize_name(value):
    if value is None:
        return ""
    return str(value).strip().casefold()


def build_city_to_country_map(world_cities_path, city_col, country_col):
    country_counter_by_city = defaultdict(Counter)

    with open(world_cities_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header row: {world_cities_path}")
        if city_col not in reader.fieldnames:
            available = ", ".join(reader.fieldnames)
            raise ValueError(
                f"Missing city column '{city_col}' in {world_cities_path}. Available: {available}"
            )
        if country_col not in reader.fieldnames:
            available = ", ".join(reader.fieldnames)
            raise ValueError(
                f"Missing country column '{country_col}' in {world_cities_path}. Available: {available}"
            )

        for row in reader:
            city = (row.get(city_col) or "").strip()
            country = (row.get(country_col) or "").strip()
            normalized_city = normalize_name(city)
            if not normalized_city or not country:
                continue
            country_counter_by_city[normalized_city][country] += 1

    city_to_country = {}
    ambiguous_city_count = 0
    for normalized_city, country_counter in country_counter_by_city.items():
        if not country_counter:
            continue
        top_country, _ = country_counter.most_common(1)[0]
        if len(country_counter) > 1:
            ambiguous_city_count += 1
        city_to_country[normalized_city] = top_country

    return city_to_country, ambiguous_city_count


def preprocess_rows(input_path, city_col, city_to_country, output_path):
    kept_rows = 0
    missing_country_rows = 0

    with open(input_path, "r", encoding="utf-8", newline="") as in_handle:
        reader = csv.DictReader(in_handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header row: {input_path}")

        required_cols = [city_col, "type", "title", "description"]
        missing_cols = [col for col in required_cols if col not in reader.fieldnames]
        if missing_cols:
            available = ", ".join(reader.fieldnames)
            raise ValueError(
                f"Missing required columns in {input_path}: {', '.join(missing_cols)}. "
                f"Available columns: {available}"
            )

        out_fieldnames = ["city", "country", "type", "title", "description"]
        with open(output_path, "w", encoding="utf-8", newline="") as out_handle:
            writer = csv.DictWriter(out_handle, fieldnames=out_fieldnames)
            writer.writeheader()

            for row in reader:
                city_value = (row.get(city_col) or "").strip()
                if not city_value:
                    continue

                normalized_city = normalize_name(city_value)
                country = city_to_country.get(normalized_city, "")
                if not country:
                    missing_country_rows += 1

                writer.writerow(
                    {
                        "city": city_value,
                        "country": country,
                        "type": (row.get("type") or "").strip(),
                        "title": (row.get("title") or "").strip(),
                        "description": (row.get("description") or "").strip(),
                    }
                )
                kept_rows += 1

    return kept_rows, missing_country_rows


def main():
    args = parse_args()

    input_path = Path(args.input_path)
    world_cities_path = Path(args.world_cities_path)
    output_path = Path(args.output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if not world_cities_path.exists():
        raise FileNotFoundError(f"World cities CSV not found: {world_cities_path}")

    city_to_country, ambiguous_city_count = build_city_to_country_map(
        world_cities_path=world_cities_path,
        city_col=args.city_column_in_world,
        country_col=args.country_column_in_world,
    )

    kept_rows, missing_country_rows = preprocess_rows(
        input_path=input_path,
        city_col=args.city_column_in_input,
        city_to_country=city_to_country,
        output_path=output_path,
    )

    print(f"City->country entries loaded: {len(city_to_country)}")
    print(f"Ambiguous city names in world cities map: {ambiguous_city_count}")
    print(f"Rows written: {kept_rows}")
    print(f"Rows with missing country: {missing_country_rows}")
    print(f"Saved preprocessed CSV to: {output_path}")


if __name__ == "__main__":
    main()

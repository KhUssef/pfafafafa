import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Replace city values in a preprocessed Wikivoyage CSV with corresponding "
            "subcountry values from world-cities.csv and save as POI dataset."
        )
    )
    parser.add_argument(
        "--input-path",
        default="wikivoyage-city-preprocessed.csv",
        help="Path to input preprocessed CSV.",
    )
    parser.add_argument(
        "--world-cities-path",
        default="world-cities.csv",
        help="Path to world cities CSV used for mapping.",
    )
    parser.add_argument(
        "--input-city-column",
        default="city",
        help="City column name in input CSV.",
    )
    parser.add_argument(
        "--input-country-column",
        default="country",
        help="Country column name in input CSV.",
    )
    parser.add_argument(
        "--world-city-column",
        default="name",
        help="City column name in world-cities CSV.",
    )
    parser.add_argument(
        "--world-country-column",
        default="country",
        help="Country column name in world-cities CSV.",
    )
    parser.add_argument(
        "--world-subcountry-column",
        default="subcountry",
        help="Subcountry column name in world-cities CSV.",
    )
    parser.add_argument(
        "--output-path",
        default="POI-dataset.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def normalize(value):
    return (value or "").strip().casefold()


def build_subcountry_map(world_cities_path, city_col, country_col, subcountry_col):
    mapping = {}
    with open(world_cities_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header row: {world_cities_path}")

        required_cols = [city_col, country_col, subcountry_col]
        missing = [col for col in required_cols if col not in reader.fieldnames]
        if missing:
            available = ", ".join(reader.fieldnames)
            raise ValueError(
                f"Missing world-cities columns: {', '.join(missing)}. Available columns: {available}"
            )

        for row in reader:
            city = normalize(row.get(city_col, ""))
            country = normalize(row.get(country_col, ""))
            subcountry = (row.get(subcountry_col) or "").strip()
            if not city or not country or not subcountry:
                continue
            # First match is kept to avoid oscillation if duplicates exist.
            mapping.setdefault((city, country), subcountry)

    return mapping


def transform_input(input_path, output_path, city_col, country_col, subcountry_map):
    total_rows = 0
    replaced_rows = 0
    missing_rows = 0

    with open(input_path, "r", encoding="utf-8", newline="") as in_handle:
        reader = csv.DictReader(in_handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header row: {input_path}")

        required_cols = [city_col, country_col]
        missing = [col for col in required_cols if col not in reader.fieldnames]
        if missing:
            available = ", ".join(reader.fieldnames)
            raise ValueError(
                f"Missing input columns: {', '.join(missing)}. Available columns: {available}"
            )

        fieldnames = list(reader.fieldnames)
        with open(output_path, "w", encoding="utf-8", newline="") as out_handle:
            writer = csv.DictWriter(out_handle, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                total_rows += 1
                city = normalize(row.get(city_col, ""))
                country = normalize(row.get(country_col, ""))
                key = (city, country)

                subcountry = subcountry_map.get(key)
                if subcountry:
                    row[city_col] = subcountry
                    replaced_rows += 1
                else:
                    missing_rows += 1

                writer.writerow(row)

    return total_rows, replaced_rows, missing_rows


def main():
    args = parse_args()

    input_path = Path(args.input_path)
    world_cities_path = Path(args.world_cities_path)
    output_path = Path(args.output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if not world_cities_path.exists():
        raise FileNotFoundError(f"World cities CSV not found: {world_cities_path}")

    subcountry_map = build_subcountry_map(
        world_cities_path=world_cities_path,
        city_col=args.world_city_column,
        country_col=args.world_country_column,
        subcountry_col=args.world_subcountry_column,
    )

    total_rows, replaced_rows, missing_rows = transform_input(
        input_path=input_path,
        output_path=output_path,
        city_col=args.input_city_column,
        country_col=args.input_country_column,
        subcountry_map=subcountry_map,
    )

    print(f"Subcountry mappings loaded: {len(subcountry_map)}")
    print(f"Rows processed: {total_rows}")
    print(f"City values replaced with subcountry: {replaced_rows}")
    print(f"Rows without mapping (city left unchanged): {missing_rows}")
    print(f"Saved POI dataset to: {output_path}")


if __name__ == "__main__":
    main()

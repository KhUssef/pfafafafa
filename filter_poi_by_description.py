import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter POI dataset to keep only rows with non-empty descriptions."
    )
    parser.add_argument(
        "--input-path",
        default="POI-dataset.csv",
        help="Path to input POI dataset CSV.",
    )
    parser.add_argument(
        "--output-path",
        default="POI-dataset-with-descriptions.csv",
        help="Path to output filtered CSV.",
    )
    parser.add_argument(
        "--description-column",
        default="description",
        help="Name of the description column.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    rows_total = 0
    rows_kept = 0

    with open(input_path, "r", encoding="utf-8", newline="") as in_handle:
        reader = csv.DictReader(in_handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header row: {input_path}")

        if args.description_column not in reader.fieldnames:
            available = ", ".join(reader.fieldnames)
            raise ValueError(
                f"Description column '{args.description_column}' not found. "
                f"Available columns: {available}"
            )

        with open(output_path, "w", encoding="utf-8", newline="") as out_handle:
            writer = csv.DictWriter(out_handle, fieldnames=reader.fieldnames)
            writer.writeheader()

            for row in reader:
                rows_total += 1
                description = (row.get(args.description_column) or "").strip()

                if description:
                    writer.writerow(row)
                    rows_kept += 1

    print(f"Total rows: {rows_total}")
    print(f"Rows with description: {rows_kept}")
    print(f"Rows removed: {rows_total - rows_kept}")
    print(f"Saved filtered dataset to: {output_path}")


if __name__ == "__main__":
    main()

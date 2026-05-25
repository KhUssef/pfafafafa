#!/usr/bin/env python3
"""Merge Wikivoyage description CSVs with city filtering.

Default behavior:
- Scans `source` CSV (default: root world-wikivoyage-descriptions-redo.csv or world-wikivoyage-descriptions.csv)
- Keeps rows whose city exists in `cities` CSV (default: new/simplemaps_worldcities_basicv1.901/worldcities.csv)
- Appends all rows from `new/world-wikivoyage-descriptions.csv`, avoiding duplicates
- Writes merged CSV to `out`

Usage: python merge_wikivoyage_descriptions.py [--source PATH] [--new PATH] [--cities PATH] [--out PATH]
"""
import argparse
import csv
import os
import unicodedata
from typing import Tuple


def normalize(text: str) -> str:
    if text is None:
        return ""
    # Normalize unicode, remove diacritics, lowercase, and strip
    text = text.strip()
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(c for c in nfkd if not unicodedata.combining(c))
    return ascii_text.casefold()


def load_cities(cities_csv_path: str) -> set:
    seen = set()
    if not os.path.exists(cities_csv_path):
        raise FileNotFoundError(cities_csv_path)
    with open(cities_csv_path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            city = row.get('city') or row.get('city_ascii') or ''
            country = row.get('country') or ''
            seen.add((normalize(city), normalize(country)))
            # also add city_ascii if present
            if 'city_ascii' in row and row['city_ascii']:
                seen.add((normalize(row['city_ascii']), normalize(country)))
    return seen


def read_rows(path: str) -> Tuple[list, list]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        rows = [row for row in reader]
        return reader.fieldnames, rows


def write_rows(path: str, fieldnames, rows: list):
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser()
    default_new = os.path.join('new', 'world-wikivoyage-descriptions.csv')
    default_cities = os.path.join('new', 'simplemaps_worldcities_basicv1.901', 'worldcities.csv')
    # choose a sensible default for source (prefer REDO file if present)
    default_source_candidates = [
        'world-wikivoyage-descriptions-redo.csv',
        'world-wikivoyage-descriptions.csv',
    ]
    default_source = None
    for c in default_source_candidates:
        if os.path.exists(c):
            default_source = c
            break
    if default_source is None:
        default_source = default_new  # fallback to new file

    parser.add_argument('--source', default=default_source, help='Source CSV to iterate (root)')
    parser.add_argument('--new', default=default_new, help='New CSV to append (new/)')
    parser.add_argument('--cities', default=default_cities, help='Cities CSV to check existence')
    parser.add_argument('--out', default='new/merged-world-wikivoyage-descriptions.csv', help='Output merged CSV')
    args = parser.parse_args()

    cities_set = load_cities(args.cities)

    # Read source and keep rows whose (city,country) exist in cities_set
    src_fields, src_rows = read_rows(args.source)
    # Ensure we have expected columns
    expected_cols = ['city', 'country', 'description']
    fieldnames = expected_cols

    kept = []
    seen_pairs = set()
    for r in src_rows:
        city = r.get('city', '')
        country = r.get('country', '')
        key = (normalize(city), normalize(country))
        if key in cities_set:
            kept.append({k: r.get(k, '') for k in fieldnames})
            seen_pairs.add(key)

    # Read new file and append any rows not already seen
    new_fields, new_rows = read_rows(args.new)
    for r in new_rows:
        city = r.get('city', '')
        country = r.get('country', '')
        key = (normalize(city), normalize(country))
        if key not in seen_pairs:
            kept.append({k: r.get(k, '') for k in fieldnames})
            seen_pairs.add(key)

    # Write merged output
    write_rows(args.out, fieldnames, kept)
    print(f'Wrote {len(kept)} rows to {args.out}')


if __name__ == '__main__':
    main()

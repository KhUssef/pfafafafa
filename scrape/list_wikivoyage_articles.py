import argparse
import csv
from pathlib import Path

DEFAULT_INPUT_CSV = Path(__file__).resolve().parent.parent / "wikivoyage-listings-en.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read Wikivoyage listings CSV and print article names."
    )
    parser.add_argument(
        "--input-csv",
        default=str(DEFAULT_INPUT_CSV),
        help="Path to wikivoyage-listings-en.csv",
    )
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="Print article values row-by-row (including duplicates).",
    )
    parser.add_argument(
        "--with-counts",
        action="store_true",
        help="Print each unique article with its listing count.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of printed lines (after header).",
    )
    parser.add_argument(
        "--sort",
        choices=["article", "article-desc", "count", "count-desc"],
        default="article",
        help="Sort mode for output. Use count/count-desc with --with-counts.",
    )
    return parser.parse_args()


def read_articles(csv_path):
    articles = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "article" not in (reader.fieldnames or []):
            raise ValueError("CSV does not contain an 'article' column")

        for row in reader:
            article = (row.get("article") or "").strip()
            if article:
                articles.append(article)

    return articles


def print_unique_articles(articles, with_counts=False, limit=None, sort_mode="article"):
    if with_counts:
        counts = {}
        for article in articles:
            counts[article] = counts.get(article, 0) + 1

        if sort_mode == "count":
            unique_sorted = sorted(counts.items(), key=lambda x: (x[1], x[0].lower(), x[0]))
        elif sort_mode == "count-desc":
            unique_sorted = sorted(counts.items(), key=lambda x: (-x[1], x[0].lower(), x[0]))
        elif sort_mode == "article-desc":
            unique_sorted = sorted(counts.items(), key=lambda x: (x[0].lower(), x[0]), reverse=True)
        else:
            unique_sorted = sorted(counts.items(), key=lambda x: (x[0].lower(), x[0]))

        print(f"Unique articles: {len(unique_sorted)}")
        rows = unique_sorted if limit is None else unique_sorted[: max(0, limit)]
        for article, count in rows:
            print(f"{article}\t{count}")
        return

    unique_articles = sorted(set(articles), key=lambda x: (x.lower(), x))
    if sort_mode in ("count", "count-desc"):
        print("Warning: --sort count/count-desc requires --with-counts; using article sort.")
        sort_mode = "article"

    if sort_mode == "article-desc":
        unique_sorted = list(reversed(unique_articles))
    else:
        unique_sorted = unique_articles

    print(f"Unique articles: {len(unique_sorted)}")
    rows = unique_sorted if limit is None else unique_sorted[: max(0, limit)]
    for article in rows:
        print(article)


def main():
    args = parse_args()
    csv_path = Path(args.input_csv)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    articles = read_articles(csv_path)

    if args.all_rows:
        print(f"Article rows: {len(articles)}")
        rows = articles if args.limit is None else articles[: max(0, args.limit)]
        for article in rows:
            print(article)
        return

    print_unique_articles(
        articles,
        with_counts=args.with_counts,
        limit=args.limit,
        sort_mode=args.sort,
    )


if __name__ == "__main__":
    main()

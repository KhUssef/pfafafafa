import argparse
import csv
import json
import time
from pathlib import Path

import faiss
import numpy as np
import requests
from tqdm import tqdm

# =========================
# CONFIG
# =========================

LM_STUDIO_URL = "http://127.0.0.1:1234/v1/embeddings"
EMBED_MODEL = "qwen/text-embedding-qwen3-embedding-0.6b"

REQUEST_TIMEOUT = (10, 120)
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
EMBED_BATCH_SIZE = 128

# Default input is the preprocessed CSV in workspace root when this script is under scrape/.
DEFAULT_INPUT_CSV = "wikivoyage-city-preprocessed.csv"

# =========================
# STORAGE
# =========================

index = None


# =========================
# TEXT / CHUNKING
# =========================


def split_lines_with_overlap(lines, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    lines = [line.strip() for line in (lines or []) if line and line.strip()]

    if not lines:
        return [""]

    chunks = []
    current_lines = []
    current_size = 0

    for line in lines:
        line_size = len(line) + 1  # +1 for joining newline separator

        # Keep very long lines intact rather than cutting them in the middle.
        if line_size > chunk_size and not current_lines:
            chunks.append(line)
            continue

        if current_lines and (current_size + line_size) > chunk_size:
            chunks.append("\n".join(current_lines))

            if overlap > 0:
                overlap_lines = []
                overlap_size = 0

                for prev in reversed(current_lines):
                    prev_size = len(prev) + 1
                    if overlap_lines and (overlap_size + prev_size) > overlap:
                        break
                    overlap_lines.append(prev)
                    overlap_size += prev_size

                overlap_lines.reverse()
                current_lines = overlap_lines
                current_size = sum(len(item) + 1 for item in current_lines)
            else:
                current_lines = []
                current_size = 0

        current_lines.append(line)
        current_size += line_size

    if current_lines:
        chunks.append("\n".join(current_lines))

    return chunks


def row_to_lines(row):
    city = (row.get("city") or "").strip()
    country = (row.get("country") or "").strip()
    listing_type = (row.get("type") or "").strip()
    title = (row.get("title") or "").strip()
    description = (row.get("description") or "").strip()

    parts = []

    name = title or "Unknown place"
    parts.append(name + ".")

    if city:
        parts.append(f"In city: {city}.")
    if country:
        parts.append(f"Country: {country}.")
    if listing_type:
        parts.append(f"Category: {listing_type}.")
    if description:
        parts.append(description)

    return parts


def chunk_row_text(row):
    lines = row_to_lines(row)
    return split_lines_with_overlap(lines)


# =========================
# I/O
# =========================


def count_csv_rows(csv_path, max_rows=None):
    if max_rows is not None and max_rows <= 0:
        return 0

    count = 0
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for _ in reader:
            count += 1
            if max_rows is not None and count >= max_rows:
                break
    return count


def iter_rows_from_csv(csv_path, max_rows=None):
    if max_rows is not None and max_rows <= 0:
        return

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            yield i, row


def estimate_totals_for_progress(csv_path, max_rows=None):
    total_rows = count_csv_rows(csv_path, max_rows=max_rows)
    total_chunks = 0

    progress_scan = tqdm(total=total_rows, desc="Estimating chunks", unit="row")
    try:
        for _, row in iter_rows_from_csv(csv_path, max_rows=max_rows):
            total_chunks += len(chunk_row_text(row))
            progress_scan.update(1)
    finally:
        progress_scan.close()

    return total_rows, total_chunks


# =========================
# EMBEDDING
# =========================


def embed_batch(texts_batch):
    response = requests.post(
        LM_STUDIO_URL,
        json={
            "model": EMBED_MODEL,
            "input": texts_batch,
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    data = response.json()
    return [np.array(x["embedding"], dtype=np.float32) for x in data["data"]]


# =========================
# PIPELINE
# =========================


def reset_storage():
    global index
    index = None


def convert_jsonl_to_json_array(jsonl_path, json_path):
    with open(jsonl_path, "r", encoding="utf-8") as src, open(json_path, "w", encoding="utf-8") as dst:
        dst.write("[\n")
        first = True

        for line in src:
            line = line.strip()
            if not line:
                continue

            if not first:
                dst.write(",\n")

            dst.write(line)
            first = False

        dst.write("\n]\n")


def save_outputs(output_prefix="wikivoyage_city", output_dir=None, *, texts_jsonl_path=None, metadata_jsonl_path=None):
    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = output_dir / f"{output_prefix}.index"
    texts_path = output_dir / f"{output_prefix}_texts.json"
    metadata_path = output_dir / f"{output_prefix}_metadata.json"

    faiss.write_index(index, str(index_path))

    if texts_jsonl_path is None or metadata_jsonl_path is None:
        raise ValueError("texts_jsonl_path and metadata_jsonl_path are required for saving outputs")

    convert_jsonl_to_json_array(texts_jsonl_path, texts_path)
    convert_jsonl_to_json_array(metadata_jsonl_path, metadata_path)

    print(f"Saved index to {index_path}")
    print(f"Saved texts to {texts_path}")
    print(f"Saved metadata to {metadata_path}")


def build_and_embed(input_csv, *, output_dir, output_prefix, max_rows=None):
    global index

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    texts_jsonl_path = output_dir / f"{output_prefix}_texts.jsonl"
    metadata_jsonl_path = output_dir / f"{output_prefix}_metadata.jsonl"

    rows_count = 0
    chunks_prepared = 0
    embedded_chunks = 0

    batch_texts = []
    batch_meta = []

    total_rows, estimated_chunks = estimate_totals_for_progress(input_csv, max_rows=max_rows)

    progress_rows = tqdm(total=total_rows, desc="Rows processed", unit="row")
    progress_embed = tqdm(total=estimated_chunks, desc="Chunks embedded", unit="chunk")

    def flush_batch(text_handle, meta_handle):
        global index
        nonlocal embedded_chunks, batch_texts, batch_meta
        if not batch_texts:
            return

        vectors = embed_batch(batch_texts)

        for text, vec, meta_item in zip(batch_texts, vectors, batch_meta):
            if index is None:
                # Initialize FAISS index on first vector dimension.
                dimension = len(vec)
                if dimension <= 0:
                    raise RuntimeError("Received empty embedding vector")
                index = faiss.IndexFlatL2(dimension)

            index.add(np.array([vec], dtype=np.float32))

            text_handle.write(json.dumps(text, ensure_ascii=False) + "\n")
            meta_handle.write(json.dumps(meta_item, ensure_ascii=False) + "\n")

        batch_size = len(batch_texts)
        embedded_chunks += batch_size
        progress_embed.update(batch_size)

        batch_texts = []
        batch_meta = []

    try:
        with open(texts_jsonl_path, "w", encoding="utf-8") as texts_f, open(metadata_jsonl_path, "w", encoding="utf-8") as meta_f:
            for row_id, row in iter_rows_from_csv(input_csv, max_rows=max_rows):
                rows_count += 1
                progress_rows.update(1)

                chunks = chunk_row_text(row)
                chunk_count = len(chunks)

                for chunk_index, text in enumerate(chunks):
                    chunks_prepared += 1

                    batch_texts.append(text)
                    batch_meta.append(
                        {
                            "row_id": row_id,
                            "city": row.get("city", ""),
                            "country": row.get("country", ""),
                            "type": row.get("type", ""),
                            "title": row.get("title", ""),
                            "chunk_index": chunk_index,
                            "chunk_count": chunk_count,
                        }
                    )

                    if len(batch_texts) >= EMBED_BATCH_SIZE:
                        flush_batch(texts_f, meta_f)

            flush_batch(texts_f, meta_f)
    finally:
        progress_rows.close()
        progress_embed.close()

    if rows_count == 0:
        print("No rows to process.")
    else:
        print(f"Loaded {rows_count} rows from {input_csv}")
        print(f"Prepared {chunks_prepared} chunks from {rows_count} CSV rows")

    return texts_jsonl_path, metadata_jsonl_path


def run_pipeline(input_csv=DEFAULT_INPUT_CSV, max_rows=None, output_prefix="POI", output_dir=None):
    reset_storage()

    started = time.time()

    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)

    texts_jsonl_path, metadata_jsonl_path = build_and_embed(
        input_csv,
        output_dir=output_dir,
        output_prefix=output_prefix,
        max_rows=max_rows,
    )

    elapsed = time.time() - started
    print(f"Total runtime: {elapsed:.2f}s")

    if index is None:
        print("No vectors were created; nothing to save.")
        return

    save_outputs(
        output_prefix=output_prefix,
        output_dir=output_dir,
        texts_jsonl_path=texts_jsonl_path,
        metadata_jsonl_path=metadata_jsonl_path,
    )

    # Remove temporary JSONL files after producing the final JSON arrays.
    texts_jsonl_path.unlink(missing_ok=True)
    metadata_jsonl_path.unlink(missing_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load preprocessed city Wikivoyage CSV data, chunk text, "
            "embed with LM Studio, and build a FAISS index with tqdm progress."
        )
    )
    parser.add_argument(
        "--input-csv",
        default=str(DEFAULT_INPUT_CSV),
        help="Path to preprocessed city CSV file (UTF-8).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row limit for quick test runs.",
    )
    parser.add_argument(
        "--output-prefix",
        default="POI",
        help="Output file prefix (<prefix>.index, <prefix>_texts.json, <prefix>_metadata.json).",
    )
    parser.add_argument(
        "--output-dir",
        default="embeddings",
        help="Optional output directory. Defaults to current working directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    run_pipeline(
        input_csv=args.input_csv,
        max_rows=args.max_rows,
        output_prefix=args.output_prefix,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

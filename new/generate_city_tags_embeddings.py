import argparse
import ast
import csv
import json
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

EMBED_URL = "http://127.0.0.1:1234/v1/embeddings"
EMBED_MODEL = "qwen/text-embedding-qwen3-embedding-0.6b"
REQUEST_TIMEOUT = (10, 180)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate embeddings and FAISS index from a pre-tagged CSV."
    )
    parser.add_argument("--input-csv", default="deepseek_tagged_cities.csv")
    parser.add_argument("--embeddings-out", default="embeddings/world-subcountries-tagged-embeddings.npy")
    parser.add_argument("--metadata-out", default="embeddings/world-subcountries-tagged-metadata.json")
    parser.add_argument("--index-out", default="embeddings/world-subcountries-tagged.index")
    parser.add_argument("--embed-url", default=EMBED_URL)
    parser.add_argument("--embed-model", default=EMBED_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-subcountries", type=int, default=0)
    parser.add_argument(
        "--test-query",
        default="",
        help="Optional query to test retrieval against generated embeddings/index.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


# ── CSV loading ───────────────────────────────────────────────────────────────

def _parse_tags(raw_tags):
    """Parse raw tag string into a list of {tag, weight} dicts."""
    raw_tags = (raw_tags or "").strip()
    if not raw_tags:
        return []

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(raw_tags)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
        if isinstance(parsed, list):
            result = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                tag = str(item.get("tag", "")).strip().lower()
                try:
                    weight = float(item.get("weight", 0))
                except (TypeError, ValueError):
                    continue
                if tag and weight > 0:
                    result.append({"tag": tag, "weight": weight})
            return result

    return []


def _tag_text_from_row(row):
    """Serialize tags back to a string for metadata storage."""
    raw_tags = (row.get("tags") or row.get("tags_json") or "").strip()
    tags = _parse_tags(raw_tags)
    if not tags:
        return ""
    parts = [f"{t['tag']}:{t['weight']:.4f}" for t in tags]
    return ", ".join(parts)


def load_tagged_rows(path):
    rows = []
    seen = set()

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"city", "tags"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

        for row in reader:
            city = (row.get("city") or "").strip()
            raw_tags = (row.get("tags") or row.get("tags_json") or "").strip()
            tags = _parse_tags(raw_tags)

            if not city or not tags:
                continue
            if city in seen:
                continue
            seen.add(city)

            rows.append({
                "city": city,
                "tags": tags,                       # structured, used for embedding
                "tag_text": _tag_text_from_row(row),  # serialized, stored in metadata
                "tourist_value": {
                    "score": (row.get("tourist_value_score") or row.get("score") or "").strip(),
                    "level": (row.get("tourist_value_level") or row.get("level") or "").strip(),
                },
            })
    return rows


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_batch(texts, embed_url, embed_model):
    resp = requests.post(
        embed_url, json={"model": embed_model, "input": texts}, timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    vectors = [np.array(item["embedding"], dtype=np.float32) for item in resp.json()["data"]]
    if len(vectors) != len(texts):
        raise RuntimeError(f"Embedding count mismatch: {len(vectors)} vs {len(texts)}")
    return vectors


def embed_all_cities_weighted(rows, embed_url, embed_model, batch_size):
    """
    Flatten all tags across all cities into one batched embedding pass,
    then reassemble per-city vectors as a weight-normalized sum.
    Result: one embedding per city that semantically reflects tag weights.
    """
    # Build flat list of (city_index, tag_text, normalized_weight)
    flat_city_idx = []
    flat_texts    = []
    flat_weights  = []

    for city_idx, row in enumerate(rows):
        tags = row["tags"]
        total_weight = sum(t["weight"] for t in tags) or 1.0
        for t in tags:
            flat_city_idx.append(city_idx)
            flat_texts.append(t["tag"])
            flat_weights.append(t["weight"] / total_weight)

    # Embed all tags in batches (single progress bar)
    all_vectors = []
    for i in tqdm(range(0, len(flat_texts), batch_size), desc="Embedding", unit="batch"):
        all_vectors.extend(
            embed_batch(flat_texts[i : i + batch_size], embed_url, embed_model)
        )

    # Accumulate weighted vectors into per-city slots
    dim = all_vectors[0].shape[0]
    city_vectors = np.zeros((len(rows), dim), dtype=np.float32)

    for vec, weight, city_idx in zip(all_vectors, flat_weights, flat_city_idx):
        city_vectors[city_idx] += vec * weight

    return city_vectors


# ── FAISS ─────────────────────────────────────────────────────────────────────

def build_faiss_index(embeddings, index_path):
    if not index_path:
        return
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("Install faiss: pip install faiss-cpu") from exc
    idx = faiss.IndexFlatL2(embeddings.shape[1])
    idx.add(embeddings)
    faiss.write_index(idx, str(index_path))


# ── Query test ────────────────────────────────────────────────────────────────

def test_query_search(query, rows, embeddings, embed_url, embed_model, top_k=5):
    query = (query or "").strip()
    if not query:
        return

    query_vector = np.vstack(embed_batch([query], embed_url, embed_model)).astype(np.float32)
    k = max(1, min(int(top_k), len(rows)))

    try:
        import faiss

        idx = faiss.IndexFlatL2(embeddings.shape[1])
        idx.add(embeddings)
        distances, indices = idx.search(query_vector, k)
        top_indices = indices[0].tolist()
        top_distances = distances[0].tolist()
    except ImportError:
        deltas = embeddings - query_vector[0]
        distances = np.sum(deltas * deltas, axis=1)
        top_indices = np.argsort(distances)[:k].tolist()
        top_distances = distances[top_indices].tolist()

    print(f"\nTest query: {query}")
    print(f"Top {k} matches:")
    for rank, (idx, dist) in enumerate(zip(top_indices, top_distances), start=1):
        item = rows[idx]
        print(
            f"{rank}. {item['city']} | "
            f"distance={float(dist):.6f} | tags={item['tag_text']}"
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    input_path = Path(args.input_csv)
    embed_path = Path(args.embeddings_out)
    meta_path  = Path(args.metadata_out)
    index_path = Path(args.index_out) if args.index_out else None

    rows = load_tagged_rows(input_path)
    if args.max_subcountries > 0:
        rows = rows[: args.max_subcountries]

    if not rows:
        raise RuntimeError("No valid rows found in input CSV.")

    # Strip internal 'tags' list before saving metadata (keep tag_text only)
    embeddings = embed_all_cities_weighted(rows, args.embed_url, args.embed_model, args.batch_size)

    np.save(embed_path, embeddings)

    meta_rows = [{k: v for k, v in r.items() if k != "tags"} for r in rows]
    meta_path.write_text(json.dumps(meta_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    build_faiss_index(embeddings, index_path)

    print(f"\nDone — {len(rows)} cities")
    print(f"Embeddings: {embed_path}  shape={embeddings.shape}")
    print(f"Metadata:   {meta_path}")
    if index_path:
        print(f"FAISS:      {index_path}")

    if args.test_query:
        test_query_search(
            query=args.test_query,
            rows=rows,
            embeddings=embeddings,
            embed_url=args.embed_url,
            embed_model=args.embed_model,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()
import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import requests

EMBED_URL = "http://127.0.0.1:1234/v1/embeddings"
EMBED_MODEL = "qwen/text-embedding-qwen3-embedding-0.6b"
REQUEST_TIMEOUT = (10, 120)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Search nearest cities with FAISS and rerank using a combined distance "
            "between semantic distance and tourist distance."
        )
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        default=["cold", "historical", "winter", "spas", "cultural"],
        help="Tags to search with (example: --tags hot historic beach).",
    )
    parser.add_argument("--index-path", default="world-subcountries-tagged.index")
    parser.add_argument("--metadata-path", default="world-subcountries-tagged-metadata.json")
    parser.add_argument("--embed-url", default=EMBED_URL)
    parser.add_argument("--embed-model", default=EMBED_MODEL)
    parser.add_argument("--top-k", type=int, default=10, help="Final number of results")
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=100,
        help="How many nearest FAISS candidates to rerank",
    )
    parser.add_argument(
        "--show-description-chars",
        type=int,
        default=180,
        help="Description preview length in output",
    )
    parser.add_argument(
        "--combine-mode",
        choices=["avg", "weighted"],
        default="avg",
        help="How to combine semantic and tourist distances.",
    )
    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=1.0,
        help="Weight for semantic distance when --combine-mode weighted.",
    )
    parser.add_argument(
        "--tourist-weight",
        type=float,
        default=1.0,
        help="Weight for tourist distance when --combine-mode weighted.",
    )
    parser.add_argument(
        "--semantic-exp",
        type=float,
        default=2.0,
        help=(
            "Exponential factor for semantic distance shaping (>=1 emphasizes high similarity)."
        ),
    )
    parser.add_argument(
        "--tourist-exp",
        type=float,
        default=1.0,
        help="Exponential factor for tourist distance shaping.",
    )
    return parser.parse_args()


def tags_to_query(tags):
    clean_tags = [str(tag).strip() for tag in (tags or []) if str(tag).strip()]
    if not clean_tags:
        return "tourism destination"
    return "tourism destination with " + ", ".join(clean_tags)


def embed_query(text, embed_url, embed_model):
    response = requests.post(
        embed_url,
        json={"model": embed_model, "input": [text]},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json().get("data", [])
    if not data:
        raise RuntimeError("No embedding returned for query")
    return np.array(data[0]["embedding"], dtype=np.float32)


def load_metadata(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return data


def distance_to_similarity(distance):
    # Index is L2-based; map distance to bounded similarity in (0, 1].
    return 1.0 / (1.0 + float(distance))


def shape_distance(distance, exponent):
    # Convex shaping in [0,1]: larger exponents reward already-similar results more.
    d = max(0.0, min(1.0, float(distance)))
    exp = max(1.0, float(exponent))
    return d**exp


def combine_distances(
    semantic_distance,
    tourist_distance,
    mode="avg",
    semantic_weight=1.0,
    tourist_weight=1.0,
):
    if mode == "weighted":
        sw = max(0.0, float(semantic_weight))
        tw = max(0.0, float(tourist_weight))
        denom = sw + tw
        if denom <= 1e-12:
            return 0.5 * (float(semantic_distance) + float(tourist_distance))
        return (sw * float(semantic_distance) + tw * float(tourist_distance)) / denom

    return 0.5 * (float(semantic_distance) + float(tourist_distance))


def get_tourist_value(item):
    tv = item.get("tourist_value") if isinstance(item, dict) else None
    if isinstance(tv, dict):
        try:
            return max(0.0, min(1.0, float(tv.get("score", 0.0))))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def has_valid_tourist_value(item):
    tv = item.get("tourist_value") if isinstance(item, dict) else None
    if not isinstance(tv, dict):
        return False
    try:
        float(tv.get("score", 0.0))
    except (TypeError, ValueError):
        return False
    return True


def search_and_rerank(
    query_vec,
    index,
    metadata,
    candidate_k,
    combine_mode,
    semantic_weight,
    tourist_weight,
    semantic_exp,
    tourist_exp,
):
    query_2d = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
    distances, indices = index.search(query_2d, candidate_k)

    valid_distances = []
    for dist, idx in zip(distances[0], indices[0]):
        idx = int(idx)
        if idx < 0 or idx >= len(metadata):
            continue
        valid_distances.append(float(dist))

    if valid_distances:
        min_dist = min(valid_distances)
        max_dist = max(valid_distances)
    else:
        min_dist = 0.0
        max_dist = 1.0

    dist_span = max(max_dist - min_dist, 1e-12)

    results = []
    skipped_missing_other_index = 0
    for dist, idx in zip(distances[0], indices[0]):
        idx = int(idx)
        if idx < 0 or idx >= len(metadata):
            skipped_missing_other_index += 1
            continue

        item = metadata[idx]
        if not has_valid_tourist_value(item):
            skipped_missing_other_index += 1
            continue

        tourist_value = get_tourist_value(item)
        semantic_distance_raw = (float(dist) - min_dist) / dist_span
        tourist_distance_raw = 1.0 - tourist_value

        semantic_distance = shape_distance(semantic_distance_raw, semantic_exp)
        tourist_distance = shape_distance(tourist_distance_raw, tourist_exp)
        combined_distance = combine_distances(
            semantic_distance=semantic_distance,
            tourist_distance=tourist_distance,
            mode=combine_mode,
            semantic_weight=semantic_weight,
            tourist_weight=tourist_weight,
        )
        similarity = distance_to_similarity(dist)
        final_score = 1.0 - combined_distance

        results.append(
            {
                "idx": idx,
                "subcountry": item.get("subcountry", ""),
                "country": item.get("country", ""),
                "description": item.get("description", ""),
                "tourist_value": tourist_value,
                "distance": float(dist),
                "semantic_distance_raw": float(semantic_distance_raw),
                "tourist_distance_raw": float(tourist_distance_raw),
                "semantic_distance": float(semantic_distance),
                "tourist_distance": float(tourist_distance),
                "combined_distance": float(combined_distance),
                "similarity": float(similarity),
                "final_score": float(final_score),
                "tags": item.get("tags", []),
            }
        )

    results.sort(key=lambda r: r["combined_distance"])
    return results, skipped_missing_other_index


def print_results(results, top_k, show_description_chars):
    top = results[:top_k]
    print(f"Returned {len(top)} results")
    for rank, r in enumerate(top, start=1):
        desc = (r["description"] or "").replace("\n", " ").strip()
        preview = desc[:show_description_chars] + ("..." if len(desc) > show_description_chars else "")
        print(
            f"#{rank} {r['subcountry']}, {r['country']} | "
            f"combined={r['combined_distance']:.6f} final={r['final_score']:.6f} "
            f"sem={r['semantic_distance']:.6f} tourist={r['tourist_distance']:.6f} "
            f"sem_raw={r['semantic_distance_raw']:.6f} tourist_raw={r['tourist_distance_raw']:.6f} "
            f"tv={r['tourist_value']:.4f} dist={r['distance']:.6f}"
        )
        if preview:
            print(f"   {preview}")


def main():
    args = parse_args()

    index_path = Path(args.index_path)
    metadata_path = Path(args.metadata_path)

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata JSON not found: {metadata_path}")

    index = faiss.read_index(str(index_path))
    metadata = load_metadata(metadata_path)

    if index.ntotal != len(metadata):
        print(
            "Warning: index/metadata size mismatch. "
            f"index.ntotal={index.ntotal}, metadata={len(metadata)}. "
            "Non-overlapping entries will be ignored."
        )

    candidate_k = min(max(1, args.candidate_k), index.ntotal)
    query = tags_to_query(args.tags)
    query_vec = embed_query(query, args.embed_url, args.embed_model)

    results, skipped_missing_other_index = search_and_rerank(
        query_vec=query_vec,
        index=index,
        metadata=metadata,
        candidate_k=candidate_k,
        combine_mode=args.combine_mode,
        semantic_weight=args.semantic_weight,
        tourist_weight=args.tourist_weight,
        semantic_exp=args.semantic_exp,
        tourist_exp=args.tourist_exp,
    )
    if skipped_missing_other_index:
        print(
            f"Ignored {skipped_missing_other_index} entries not present in both semantic and tourist data."
        )
    print_results(results, args.top_k, args.show_description_chars)


if __name__ == "__main__":
    main()

import argparse
import csv
import json
import math
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
            "Rank nearest cities with a global score combining description, POIs, and tags. "
            "Tag score is multiplied by tourist value."
        )
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        default=["cold", "historical", "winter", "spas", "cultural"],
        help="Input tags to search city preferences.",
    )
    parser.add_argument("--index-path", default="world-subcountries-tagged.index")
    parser.add_argument("--cities-csv", default="world-subcountries-tagged.csv")
    parser.add_argument("--poi-csv", default="POI-dataset-with-descriptions.csv")
    parser.add_argument("--embed-url", default=EMBED_URL)
    parser.add_argument("--embed-model", default=EMBED_MODEL)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=120)
    parser.add_argument("--show-description-chars", type=int, default=180)
    parser.add_argument("--description-weight", type=float, default=1.0)
    parser.add_argument("--poi-weight", type=float, default=1.0)
    parser.add_argument("--tags-weight", type=float, default=1.0)
    return parser.parse_args()


def normalize_key(text):
    return str(text or "").strip().lower()


def build_query_text(tags):
    clean_tags = [str(tag).strip() for tag in (tags or []) if str(tag).strip()]
    if not clean_tags:
        return "tourism destination"
    return "tourism destination with " + ", ".join(clean_tags)


def build_tags_text(tags):
    clean_tags = [str(tag).strip() for tag in (tags or []) if str(tag).strip()]
    return ", ".join(clean_tags)


def embed_texts(texts, embed_url, embed_model):
    safe_texts = [str(text or "") for text in texts]
    response = requests.post(
        embed_url,
        json={"model": embed_model, "input": safe_texts},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json().get("data", [])
    if len(data) != len(safe_texts):
        raise RuntimeError(
            f"Embedding API returned {len(data)} vectors for {len(safe_texts)} inputs"
        )
    vectors = [np.array(row["embedding"], dtype=np.float32) for row in data]
    return vectors


def parse_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def parse_tags_json(tags_json_raw):
    if not tags_json_raw:
        return []
    try:
        tags = json.loads(tags_json_raw)
    except json.JSONDecodeError:
        return []

    out = []
    if isinstance(tags, list):
        for item in tags:
            if not isinstance(item, dict):
                continue
            tag_name = str(item.get("tag", "")).strip()
            tag_weight = parse_float(item.get("weight", 0.0), 0.0)
            if tag_name:
                out.append({"tag": tag_name, "weight": max(0.0, min(1.0, tag_weight))})
    return out


def load_cities_metadata(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            subcountry = str(row.get("subcountry", "")).strip()
            city = str(row.get("city", "")).strip()
            country = str(row.get("country", "")).strip()
            description = str(row.get("description", "") or "").strip()
            tourist_value = max(0.0, min(1.0, parse_float(row.get("tourist_value_score", 0.0), 0.0)))
            tags_json = parse_tags_json(row.get("tags_json"))
            tag_text = str(row.get("tag_text", "") or "").strip()

            primary_name = subcountry or city
            rows.append(
                {
                    "subcountry": subcountry,
                    "city": city,
                    "country": country,
                    "name": primary_name,
                    "description": description,
                    "tourist_value": tourist_value,
                    "tags_json": tags_json,
                    "tag_text": tag_text,
                }
            )

    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    return rows


def build_poi_lookup(poi_csv_path):
    poi_lookup = {}

    with open(poi_csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            city = normalize_key(row.get("city", ""))
            country = normalize_key(row.get("country", ""))
            title = str(row.get("title", "") or "").strip()
            description = str(row.get("description", "") or "").strip()

            if not city:
                continue

            text = f"{title}. {description}".strip().strip(".").strip()
            if not text:
                continue

            key_city = (city, "")
            key_city_country = (city, country)
            poi_lookup.setdefault(key_city, []).append(text)
            poi_lookup.setdefault(key_city_country, []).append(text)

    return poi_lookup


def get_city_poi_text(city_row, poi_lookup, max_items=25):
    city_key = normalize_key(city_row.get("name", ""))
    country_key = normalize_key(city_row.get("country", ""))

    pois = poi_lookup.get((city_key, country_key))
    if not pois:
        pois = poi_lookup.get((city_key, ""), [])

    if not pois:
        return ""

    joined = " ".join(pois[:max_items]).strip()
    return joined[:5000]


def l2_distance(vec_a, vec_b):
    return float(np.linalg.norm(np.asarray(vec_a, dtype=np.float32) - np.asarray(vec_b, dtype=np.float32)))


def cosine_distance(vec_a, vec_b):
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 1.0
    cosine_sim = float(np.dot(a, b) / denom)
    cosine_sim = max(-1.0, min(1.0, cosine_sim))
    return 1.0 - ((cosine_sim + 1.0) / 2.0)


def normalize_distance(distance, min_distance, max_distance):
    span = max(max_distance - min_distance, 1e-12)
    d = (float(distance) - float(min_distance)) / span
    return max(0.0, min(1.0, d))


def distance_to_score(normalized_distance_value):
    d = max(0.0, min(1.0, float(normalized_distance_value)))
    return 1.0 - d


def calculate_description_score(faiss_distance, min_faiss_distance, max_faiss_distance):
    normalized = normalize_distance(faiss_distance, min_faiss_distance, max_faiss_distance)
    return distance_to_score(normalized), normalized


def calculate_poi_score(query_embedding, poi_embedding):
    if poi_embedding is None:
        return 0.0, 1.0
    poi_distance = cosine_distance(query_embedding, poi_embedding)
    poi_distance = max(0.0, min(1.0, poi_distance))
    return distance_to_score(poi_distance), poi_distance


def calculate_tags_score(query_tags_embedding, city_tags_embedding, tourist_value):
    if city_tags_embedding is None:
        return 0.0, 1.0
    tags_distance = cosine_distance(query_tags_embedding, city_tags_embedding)
    tags_distance = max(0.0, min(1.0, tags_distance))
    base_tags_score = distance_to_score(tags_distance)
    weighted_tags_score = base_tags_score * max(0.0, min(1.0, float(tourist_value)))
    return weighted_tags_score, tags_distance


def calculate_final_score(
    description_score,
    poi_score,
    tags_score,
    description_weight=1.0,
    poi_weight=1.0,
    tags_weight=1.0,
):
    dw = max(0.0, float(description_weight))
    pw = max(0.0, float(poi_weight))
    tw = max(0.0, float(tags_weight))
    denom = dw + pw + tw
    if denom <= 1e-12:
        return (float(description_score) + float(poi_score) + float(tags_score)) / 3.0
    return (dw * float(description_score) + pw * float(poi_score) + tw * float(tags_score)) / denom


def build_city_tags_text(city_row):
    tags_json = city_row.get("tags_json") or []
    if tags_json:
        parts = [f"{item.get('tag', '')}:{parse_float(item.get('weight', 0.0), 0.0):.4f}" for item in tags_json]
        text = ", ".join([p for p in parts if p.strip(":")])
        if text:
            return text
    return str(city_row.get("tag_text", "") or "")


def rank_cities(args):
    index_path = Path(args.index_path)
    cities_csv = Path(args.cities_csv)
    poi_csv = Path(args.poi_csv)

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not cities_csv.exists():
        raise FileNotFoundError(f"Cities CSV not found: {cities_csv}")
    if not poi_csv.exists():
        raise FileNotFoundError(f"POI CSV not found: {poi_csv}")

    index = faiss.read_index(str(index_path))
    metadata = load_cities_metadata(cities_csv)
    poi_lookup = build_poi_lookup(poi_csv)

    if index.ntotal != len(metadata):
        print(
            "Warning: index/metadata size mismatch. "
            f"index.ntotal={index.ntotal}, metadata={len(metadata)}. Using overlapping range only."
        )

    usable_total = min(index.ntotal, len(metadata))
    if usable_total <= 0:
        raise RuntimeError("No overlapping city entries between index and metadata.")

    query_text = build_query_text(args.tags)
    tags_query_text = build_tags_text(args.tags)

    query_embedding, query_tags_embedding = embed_texts(
        [query_text, tags_query_text], args.embed_url, args.embed_model
    )

    candidate_k = min(max(1, args.candidate_k), usable_total)
    distances, indices = index.search(np.asarray(query_embedding, dtype=np.float32).reshape(1, -1), candidate_k)

    valid_faiss_distances = []
    candidate_rows = []
    for faiss_dist, idx in zip(distances[0], indices[0]):
        idx = int(idx)
        if idx < 0 or idx >= usable_total:
            continue
        row = metadata[idx]
        valid_faiss_distances.append(float(faiss_dist))
        candidate_rows.append((idx, float(faiss_dist), row))

    if not candidate_rows:
        return []

    min_faiss_dist = min(valid_faiss_distances)
    max_faiss_dist = max(valid_faiss_distances)

    poi_texts = [get_city_poi_text(row, poi_lookup) for (_, _, row) in candidate_rows]
    tags_texts = [build_city_tags_text(row) for (_, _, row) in candidate_rows]

    poi_vectors = []
    tags_vectors = []

    non_empty_poi_positions = [i for i, text in enumerate(poi_texts) if text.strip()]
    if non_empty_poi_positions:
        poi_batch = [poi_texts[i] for i in non_empty_poi_positions]
        poi_embeds = embed_texts(poi_batch, args.embed_url, args.embed_model)
        poi_vectors = [None] * len(poi_texts)
        for pos, vec in zip(non_empty_poi_positions, poi_embeds):
            poi_vectors[pos] = vec
    else:
        poi_vectors = [None] * len(poi_texts)

    non_empty_tags_positions = [i for i, text in enumerate(tags_texts) if text.strip()]
    if non_empty_tags_positions:
        tags_batch = [tags_texts[i] for i in non_empty_tags_positions]
        tag_embeds = embed_texts(tags_batch, args.embed_url, args.embed_model)
        tags_vectors = [None] * len(tags_texts)
        for pos, vec in zip(non_empty_tags_positions, tag_embeds):
            tags_vectors[pos] = vec
    else:
        tags_vectors = [None] * len(tags_texts)

    ranked = []
    for i, (idx, faiss_dist, row) in enumerate(candidate_rows):
        description_score, description_distance_norm = calculate_description_score(
            faiss_distance=faiss_dist,
            min_faiss_distance=min_faiss_dist,
            max_faiss_distance=max_faiss_dist,
        )

        poi_score, poi_distance = calculate_poi_score(query_embedding, poi_vectors[i])

        tourist_value = parse_float(row.get("tourist_value", 0.0), 0.0)
        tags_score, tags_distance = calculate_tags_score(
            query_tags_embedding=query_tags_embedding,
            city_tags_embedding=tags_vectors[i],
            tourist_value=tourist_value,
        )

        final_score = calculate_final_score(
            description_score=description_score,
            poi_score=poi_score,
            tags_score=tags_score,
            description_weight=args.description_weight,
            poi_weight=args.poi_weight,
            tags_weight=args.tags_weight,
        )

        ranked.append(
            {
                "idx": idx,
                "subcountry": row.get("subcountry", "") or row.get("name", ""),
                "city": row.get("city", "") or row.get("name", ""),
                "country": row.get("country", ""),
                "description": row.get("description", ""),
                "tourist_value": tourist_value,
                "description_score": float(description_score),
                "poi_score": float(poi_score),
                "tags_score": float(tags_score),
                "description_distance": float(description_distance_norm),
                "poi_distance": float(poi_distance),
                "tags_distance": float(tags_distance),
                "final_score": float(final_score),
                "faiss_distance": float(faiss_dist),
            }
        )

    ranked.sort(key=lambda r: r["final_score"], reverse=True)
    return ranked


def print_results(results, top_k, show_description_chars):
    top = results[: max(1, int(top_k))]
    print(f"Returned {len(top)} results")
    for rank, row in enumerate(top, start=1):
        desc = (row.get("description") or "").replace("\n", " ").strip()
        preview = desc[:show_description_chars] + ("..." if len(desc) > show_description_chars else "")
        print(
            f"#{rank} {row['subcountry']}, {row['country']} | "
            f"final={row['final_score']:.6f} "
            f"desc={row['description_score']:.6f} poi={row['poi_score']:.6f} tags={row['tags_score']:.6f} "
            f"tv={row['tourist_value']:.4f} "
            f"d_desc={row['description_distance']:.6f} d_poi={row['poi_distance']:.6f} d_tags={row['tags_distance']:.6f}"
        )
        if preview:
            print(f"   {preview}")


def main():
    args = parse_args()
    results = rank_cities(args)
    print_results(results, args.top_k, args.show_description_chars)


if __name__ == "__main__":
    main()

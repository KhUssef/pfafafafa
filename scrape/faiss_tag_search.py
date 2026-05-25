import argparse
import json
import math
from pathlib import Path

import faiss
import numpy as np
import requests

LM_STUDIO_URL = "http://127.0.0.1:1234/v1/embeddings"
EMBED_MODEL = "qwen/text-embedding-qwen3-embedding-0.6b"
REQUEST_TIMEOUT = (10, 120)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Embed tourism tags with LM Studio and search top matches in a FAISS index."
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        default=["cold", "relaxing", "winter", "spas", "cultural"],
        help="Tags to search with (example: --tags hot historic beach).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of final merged results to print.",
    )
    parser.add_argument(
        "--per-tag-k",
        type=int,
        default=100,
        help="How many candidates to fetch per tag before merging.",
    )
    parser.add_argument(
        "--index-path",
        default="wikivoyage.index",
        help="Path to FAISS index file.",
    )
    parser.add_argument(
        "--texts-path",
        default="wikivoyage_texts.json",
        help="Path to JSON list of chunk texts.",
    )
    parser.add_argument(
        "--metadata-path",
        default="wikivoyage_metadata.json",
        help="Path to JSON list of metadata objects.",
    )
    parser.add_argument(
        "--embed-url",
        default=LM_STUDIO_URL,
        help="LM Studio embeddings endpoint URL.",
    )
    parser.add_argument(
        "--embed-model",
        default=EMBED_MODEL,
        help="Embedding model name served by LM Studio.",
    )
    parser.add_argument(
        "--show-chars",
        type=int,
        default=260,
        help="How many characters of each text chunk to print.",
    )
    parser.add_argument(
        "--tag-weights",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional weights aligned with --tags. "
            "Provide one value to broadcast to all tags or one per tag."
        ),
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional file path to save merged results as JSON.",
    )
    parser.add_argument(
        "--article-avg",
        action="store_true",
        help="Compute article-level average distance across all listings in each article.",
    )
    parser.add_argument(
        "--article-top-k",
        type=int,
        default=20,
        help="How many top articles to return when --article-avg is enabled.",
    )
    parser.add_argument(
        "--article-entry-top-k",
        type=int,
        default=5,
        help="How many top entries to return for each selected article.",
    )
    parser.add_argument(
        "--count-reward-alpha",
        type=float,
        default=0.3,
        help="Strength of article count reward in weighted distance (higher rewards more listings).",
    )
    parser.add_argument(
        "--count-reward-tau",
        type=float,
        default=40.0,
        help="Saturation speed for count reward (smaller values saturate faster).",
    )
    return parser.parse_args()


def embed_texts(texts, embed_url, embed_model):
    response = requests.post(
        embed_url,
        json={"model": embed_model, "input": texts},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    data = response.json()
    vectors = [np.array(item["embedding"], dtype=np.float32) for item in data["data"]]

    if len(vectors) != len(texts):
        raise RuntimeError(
            f"Embedding count mismatch: got {len(vectors)} vectors for {len(texts)} inputs"
        )

    return vectors


def load_json_array(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return data


def tag_to_query(tag):
    clean = (tag or "").strip()
    if not clean:
        return "tourism place"
    return f"tourism destination that is {clean}"


def build_tag_weights(tags, raw_weights):
    if raw_weights is None:
        return np.ones(len(tags), dtype=np.float32)

    if len(raw_weights) == 1 and len(tags) > 1:
        raw_weights = raw_weights * len(tags)

    if len(raw_weights) != len(tags):
        raise ValueError(
            "--tag-weights must contain either 1 value or exactly one value per tag"
        )

    weights = np.array(raw_weights, dtype=np.float32)
    if np.any(weights <= 0):
        raise ValueError("All --tag-weights values must be > 0")
    return weights


def merge_results(index, tag_results, texts, metadata, tag_weight_by_tag):
    merged = {}

    for tag, distances, indices in tag_results:
        weight = float(tag_weight_by_tag.get(tag, 1.0))
        for distance, idx in zip(distances, indices):
            idx = int(idx)
            if idx < 0:
                continue

            distance = float(distance)
            weighted_distance = distance / max(weight, 1e-12)
            score = -weighted_distance
            entry = merged.get(idx)

            if entry is None:
                merged[idx] = {
                    "idx": idx,
                    "best_distance": distance,
                    "best_weighted_distance": weighted_distance,
                    "best_score": score,
                    "matched_tags": [tag],
                    "matched_weight_sum": weight,
                    "text": texts[idx] if idx < len(texts) else "",
                    "metadata": metadata[idx] if idx < len(metadata) else {},
                }
            else:
                if distance < entry["best_distance"]:
                    entry["best_distance"] = distance
                if weighted_distance < entry["best_weighted_distance"]:
                    entry["best_weighted_distance"] = weighted_distance
                    entry["best_score"] = score
                if tag not in entry["matched_tags"]:
                    entry["matched_tags"].append(tag)
                    entry["matched_weight_sum"] += weight

    merged_list = list(merged.values())
    merged_list.sort(
        key=lambda item: (
            -item["matched_weight_sum"],
            item["best_weighted_distance"],
            item["best_distance"],
        )
    )
    return merged_list


def print_results(results, top_k, show_chars):
    print(f"Merged results: {min(top_k, len(results))}/{len(results)}")
    for rank, item in enumerate(results[:top_k], start=1):
        meta = item["metadata"] or {}
        title = meta.get("title", "")
        article = meta.get("article", "")
        listing_type = meta.get("type", "")
        tags = ", ".join(item["matched_tags"])

        text = (item["text"] or "").replace("\n", " ").strip()
        preview = text[:show_chars]
        if len(text) > show_chars:
            preview += "..."

        print()
        print(
            f"#{rank} idx={item['idx']} dist={item['best_distance']:.4f} "
            f"wdist={item['best_weighted_distance']:.4f} "
            f"weight_sum={item['matched_weight_sum']:.2f} tags=[{tags}]"
        )
        print(f"   title={title} | article={article} | type={listing_type}")
        print(f"   text={preview}")


def build_article_groups(metadata):
    article_to_indices = {}
    for idx, meta in enumerate(metadata):
        article = (meta or {}).get("article", "")
        article = article.strip() if isinstance(article, str) else ""
        if not article:
            continue
        article_to_indices.setdefault(article, []).append(idx)
    return article_to_indices


def calc_entry_distance(vector, query_vectors, tag_weights):
    # Entry distance is weighted average squared-L2 across all tag query vectors.
    dists = np.sum((query_vectors - vector) ** 2, axis=1)
    weight_sum = float(np.sum(tag_weights))
    weighted = float(np.dot(dists, tag_weights) / max(weight_sum, 1e-12))
    return weighted, [float(x) for x in dists]


def summarize_articles(
    index,
    metadata,
    texts,
    query_vectors,
    tag_weights,
    article_top_k,
    article_entry_top_k,
    count_reward_alpha,
    count_reward_tau,
):
    article_to_indices = build_article_groups(metadata)

    summaries = []
    for article, listing_indices in article_to_indices.items():
        if not listing_indices:
            continue

        entry_distances = []
        per_tag_accumulator = np.zeros(query_vectors.shape[0], dtype=np.float64)
        article_entries = []
        for idx in listing_indices:
            if idx >= index.ntotal:
                continue
            vec = np.array(index.reconstruct(int(idx)), dtype=np.float32)
            entry_distance, tag_distances = calc_entry_distance(vec, query_vectors, tag_weights)
            entry_distances.append(entry_distance)
            per_tag_accumulator += np.array(tag_distances, dtype=np.float64)

            meta = metadata[idx] if idx < len(metadata) else {}
            article_entries.append(
                {
                    "idx": int(idx),
                    "entry_distance": float(entry_distance),
                    "per_tag_distance": [float(x) for x in tag_distances],
                    "metadata": meta,
                    "text": texts[idx] if idx < len(texts) else "",
                }
            )

        if not entry_distances:
            continue

        listing_count = len(entry_distances)
        mean_entry_distance = float(np.mean(entry_distances))
        tau = max(1e-6, float(count_reward_tau))
        # Saturating reward: additional listings still help, but with diminishing impact.
        reward = 1.0 - math.exp(-listing_count / tau)
        count_bonus = 1.0 + max(0.0, count_reward_alpha) * reward
        weighted_distance = mean_entry_distance / count_bonus
        mean_per_tag_distance = (per_tag_accumulator / listing_count).tolist()
        article_entries.sort(key=lambda item: item["entry_distance"])

        summaries.append(
            {
                "article": article,
                "weighted_distance": weighted_distance,
                "mean_entry_distance": mean_entry_distance,
                "min_entry_distance": float(np.min(entry_distances)),
                "max_entry_distance": float(np.max(entry_distances)),
                "listing_count": listing_count,
                "count_bonus": count_bonus,
                "mean_per_tag_distance": [float(x) for x in mean_per_tag_distance],
                "top_entries": article_entries[:article_entry_top_k],
            }
        )

    summaries.sort(key=lambda item: item["weighted_distance"])
    return summaries[:article_top_k], summaries


def print_article_summaries(article_summaries, show_chars):
    print("\nArticle average distance results:")
    if not article_summaries:
        print("No article summaries available.")
        return

    for rank, item in enumerate(article_summaries, start=1):
        print(
            f"#{rank} article={item['article']} weighted={item['weighted_distance']:.4f} "
            f"mean={item['mean_entry_distance']:.4f} min={item['min_entry_distance']:.4f} "
            f"max={item['max_entry_distance']:.4f} listings={item['listing_count']} "
            f"bonus={item['count_bonus']:.4f}"
        )
        for entry_rank, entry in enumerate(item.get("top_entries", []), start=1):
            meta = entry.get("metadata") or {}
            title = meta.get("title", "")
            listing_type = meta.get("type", "")
            text = (entry.get("text") or "").replace("\n", " ").strip()
            preview = text[:show_chars]
            if len(text) > show_chars:
                preview += "..."
            print(
                f"   - entry#{entry_rank} idx={entry['idx']} dist={entry['entry_distance']:.4f} "
                f"title={title} type={listing_type}"
            )
            print(f"     text={preview}")


def main():
    args = parse_args()

    index_path = Path(args.index_path)
    texts_path = Path(args.texts_path)
    metadata_path = Path(args.metadata_path)

    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")
    if not texts_path.exists():
        raise FileNotFoundError(f"Texts JSON not found: {texts_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata JSON not found: {metadata_path}")

    index = faiss.read_index(str(index_path))
    texts = load_json_array(texts_path)
    metadata = load_json_array(metadata_path)

    if len(texts) != index.ntotal:
        print(
            f"Warning: texts count ({len(texts)}) != index vectors ({index.ntotal}). "
            "Results will still run by vector index."
        )

    queries = [tag_to_query(tag) for tag in args.tags]
    tag_weights = build_tag_weights(args.tags, args.tag_weights)
    tag_weight_by_tag = {
        tag: float(weight) for tag, weight in zip(args.tags, tag_weights.tolist())
    }

    vectors = embed_texts(queries, args.embed_url, args.embed_model)
    query_vectors = np.array(vectors, dtype=np.float32)

    tag_results = []
    for tag, vec in zip(args.tags, vectors):
        distances, indices = index.search(np.array([vec], dtype=np.float32), args.per_tag_k)
        tag_results.append((tag, distances[0], indices[0]))

    merged = merge_results(index, tag_results, texts, metadata, tag_weight_by_tag)
    if not args.article_avg:
        print_results(merged, args.top_k, args.show_chars)

    article_summaries = []
    full_article_summaries = []
    if args.article_avg:
        article_summaries, full_article_summaries = summarize_articles(
            index=index,
            metadata=metadata,
            texts=texts,
            query_vectors=query_vectors,
            tag_weights=tag_weights,
            article_top_k=args.article_top_k,
            article_entry_top_k=args.article_entry_top_k,
            count_reward_alpha=args.count_reward_alpha,
            count_reward_tau=args.count_reward_tau,
        )
        print_article_summaries(article_summaries, args.show_chars)

    if args.json_output:
        output_path = Path(args.json_output)
        with open(output_path, "w", encoding="utf-8") as handle:
            payload = {
                "listing_results": merged[: args.top_k] if not args.article_avg else [],
                "article_results": article_summaries if args.article_avg else [],
                "article_results_all": full_article_summaries if args.article_avg else [],
            }
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print(f"\nSaved JSON results to {output_path}")


if __name__ == "__main__":
    main()

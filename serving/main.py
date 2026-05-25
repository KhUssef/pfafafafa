from __future__ import annotations

import ast
import csv
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import threading
import time
import uuid

import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

ARTIFACTS_DIR = ROOT_DIR / "new" / "embeddings"
DEFAULT_INDEX_PATH = ARTIFACTS_DIR / "world-subcountries-tagged.index"
DEFAULT_EMBEDDINGS_PATH = ARTIFACTS_DIR / "world-subcountries-tagged-embeddings.npy"
DEFAULT_CITY_METADATA_PATH = ARTIFACTS_DIR / "world-subcountries-tagged-metadata.json"
DEFAULT_CITY_DESCRIPTION_METADATA_PATH = ARTIFACTS_DIR / "embeddings_metadata.json"
DEFAULT_POI_INDEX_PATH = ARTIFACTS_DIR / "POI.index"
DEFAULT_POI_METADATA_PATH = ARTIFACTS_DIR / "POI_metadata.json"
DEFAULT_POI_TEXTS_PATH = ARTIFACTS_DIR / "POI_texts.json"
DEFAULT_DEEPSEEK_TAGS_CSV = ROOT_DIR / "new" / "deepseek_tagged_cities.csv"
DEFAULT_FILTERED_CITIES_DESCRIPTIONS_CSV = ROOT_DIR / "new" / "filtered_cities_with_descriptions.csv"
DEFAULT_WIKIVOYAGE_PREPROCESSED_CSV = ROOT_DIR / "new" / "wikivoyage-city-preprocessed.csv"
BOOKING_JOBS_DIR = Path(__file__).resolve().parent / "jobs"

EMBED_URL = os.getenv("EMBED_URL", "http://127.0.0.1:1234/v1/embeddings")
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen/text-embedding-qwen3-embedding-0.6b")
REQUEST_TIMEOUT = (10, 120)


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def normalize_key(text: Any) -> str:
    return str(text or "").strip().lower()


def parse_tags_json(tags_json_raw: Any) -> List[Dict[str, float]]:
    if not tags_json_raw:
        return []
    try:
        tags = json.loads(tags_json_raw)
    except json.JSONDecodeError:
        return []

    out: List[Dict[str, float]] = []
    if isinstance(tags, list):
        for item in tags:
            if not isinstance(item, dict):
                continue
            tag_name = str(item.get("tag", "")).strip().lower()
            tag_weight = parse_float(item.get("weight", 0.0), 0.0)
            if tag_name and tag_weight > 0:
                out.append({"tag": tag_name, "weight": tag_weight})
    return out


def normalize_input_tags(raw_tags: Any) -> List[Dict[str, float]]:
    if raw_tags is None:
        return []

    tags: List[Dict[str, float]] = []

    if isinstance(raw_tags, list):
        candidates = raw_tags
    elif isinstance(raw_tags, dict):
        candidates = [raw_tags]
    else:
        return []

    for item in candidates:
        if not isinstance(item, dict):
            continue

        tag_name = ""
        tag_weight = 0.0

        if "tag" in item or "weight" in item:
            tag_name = str(item.get("tag", "")).strip().lower()
            tag_weight = parse_float(item.get("weight", 0.0), 0.0)
        elif len(item) == 1:
            key, value = next(iter(item.items()))
            tag_name = str(key).strip().lower()
            tag_weight = parse_float(value, 0.0)
        else:
            continue

        if tag_name and tag_weight > 0:
            tags.append({"tag": tag_name, "weight": tag_weight})

    return tags


def build_query_text(tags: Sequence[Dict[str, float]]) -> str:
    clean_tags = [str(tag.get("tag", "")).strip() for tag in tags if str(tag.get("tag", "")).strip()]
    if not clean_tags:
        return "tourism destination"
    return "tourism destination with " + ", ".join(clean_tags)


def build_tags_text(tags: Sequence[Dict[str, float]]) -> str:
    clean_tags = [str(tag.get("tag", "")).strip() for tag in tags if str(tag.get("tag", "")).strip()]
    return ", ".join(clean_tags)


def embed_texts(texts: Sequence[str], embed_url: str, embed_model: str) -> List[np.ndarray]:
    safe_texts = [str(text or "") for text in texts]
    response = requests.post(
        embed_url,
        json={"model": embed_model, "input": safe_texts},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json().get("data", [])
    if len(data) != len(safe_texts):
        raise RuntimeError(f"Embedding API returned {len(data)} vectors for {len(safe_texts)} inputs")
    return [np.array(row["embedding"], dtype=np.float32) for row in data]


def embed_weighted_tags(
    tags: Sequence[Dict[str, float]],
    embed_url: str,
    embed_model: str,
    batch_size: int = 64,
) -> np.ndarray:
    clean_tags = [tag for tag in tags if str(tag.get("tag", "")).strip() and parse_float(tag.get("weight", 0.0), 0.0) > 0]
    if not clean_tags:
        raise ValueError("At least one valid tag with a positive weight is required.")

    flat_texts: List[str] = []
    flat_weights: List[float] = []
    total_weight = sum(parse_float(tag.get("weight", 0.0), 0.0) for tag in clean_tags) or 1.0

    for tag in clean_tags:
        flat_texts.append(str(tag.get("tag", "")).strip().lower())
        flat_weights.append(parse_float(tag.get("weight", 0.0), 0.0) / total_weight)

    vectors: List[np.ndarray] = []
    for start in range(0, len(flat_texts), max(1, int(batch_size))):
        vectors.extend(embed_texts(flat_texts[start : start + batch_size], embed_url, embed_model))

    if not vectors:
        raise RuntimeError("Embedding service returned no vectors.")

    result = np.zeros(vectors[0].shape[0], dtype=np.float32)
    for vector, weight in zip(vectors, flat_weights):
        result += vector * weight
    return result


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return [item for item in data if isinstance(item, dict)]


def load_tourist_value_lookup(csv_path: Path) -> Dict[Tuple[str, str], float]:
    lookup: Dict[Tuple[str, str], float] = {}
    if not csv_path.exists():
        return lookup

    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            city = normalize_key(row.get("city", ""))
            country = normalize_key(row.get("country", ""))

            # Primary source: deepseek tags CSV with a Python-like list in the "tags" column.
            tourist_value = 0.0
            tags_raw = row.get("tags", "")
            if tags_raw:
                try:
                    parsed_tags = ast.literal_eval(str(tags_raw))
                    if isinstance(parsed_tags, list):
                        for item in parsed_tags:
                            if not isinstance(item, dict):
                                continue
                            tag_name = str(item.get("tag", "")).strip().lower()
                            if tag_name == "touristic-value":
                                tourist_value = max(0.0, min(1.0, parse_float(item.get("weight", 0.0), 0.0)))
                                break
                except (ValueError, SyntaxError):
                    tourist_value = 0.0

            # Backward-compatible fallback for files that already expose a score column.
            if tourist_value <= 0.0:
                tourist_value = max(0.0, min(1.0, parse_float(row.get("tourist_value_score", 0.0), 0.0)))

            if city:
                lookup[(city, country)] = tourist_value
                if country:
                    lookup[(city, "")] = tourist_value
    return lookup


def load_city_description_lookup(filtered_csv_path: Path, wikivoyage_csv_path: Path) -> Dict[Tuple[str, str], str]:
    lookup: Dict[Tuple[str, str], str] = {}

    # Highest priority: one clean description per city from filtered CSV.
    if filtered_csv_path.exists():
        with open(filtered_csv_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                city = normalize_key(row.get("city", ""))
                country = normalize_key(row.get("country", ""))
                description = str(row.get("description", "") or "").strip()
                if city and description:
                    lookup[(city, country)] = description

    # Fallback: aggregate POI rows from wikivoyage preprocessed file by city/country.
    if wikivoyage_csv_path.exists():
        aggregated: Dict[Tuple[str, str], List[str]] = {}
        with open(wikivoyage_csv_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                city = normalize_key(row.get("city", ""))
                country = normalize_key(row.get("country", ""))
                if not city:
                    continue
                text = str(row.get("description", "") or "").strip()
                if text:
                    aggregated.setdefault((city, country), []).append(text)

        for key, parts in aggregated.items():
            if key in lookup:
                continue
            joined = " ".join(parts[:25]).strip()
            if joined:
                lookup[key] = joined[:5000]

    return lookup


def load_preembedded_city_metadata(
    city_metadata_path: Path,
    city_description_metadata_path: Path,
    deepseek_tags_csv: Path,
    filtered_cities_descriptions_csv: Path,
    wikivoyage_preprocessed_csv: Path,
    embeddings_path: Path,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    city_meta = load_json_list(city_metadata_path)
    desc_meta = load_json_list(city_description_metadata_path)
    embeddings = np.load(embeddings_path).astype(np.float32)

    if len(city_meta) != len(desc_meta) or len(city_meta) != len(embeddings):
        raise ValueError(
            "Preembedded city artifact sizes do not match: "
            f"tags={len(city_meta)}, descriptions={len(desc_meta)}, embeddings={len(embeddings)}"
        )

    tourist_lookup = load_tourist_value_lookup(deepseek_tags_csv)
    description_lookup = load_city_description_lookup(
        filtered_csv_path=filtered_cities_descriptions_csv,
        wikivoyage_csv_path=wikivoyage_preprocessed_csv,
    )
    rows: List[Dict[str, Any]] = []

    for tag_row, desc_row in zip(city_meta, desc_meta):
        city = str(tag_row.get("city", desc_row.get("city", ""))).strip()
        country = str(desc_row.get("country", tag_row.get("country", ""))).strip()
        description_key = (normalize_key(city), normalize_key(country))
        description = description_lookup.get(description_key) or str(desc_row.get("description", "") or "").strip()
        tag_text = str(tag_row.get("tag_text", "") or "").strip()
        tourist_value = tourist_lookup.get(description_key, tourist_lookup.get((normalize_key(city), ""), 0.0))

        rows.append(
            {
                "city": city,
                "country": country,
                "description": description,
                "tag_text": tag_text,
                "tourist_value": tourist_value,
            }
        )

    return rows, embeddings


def load_preembedded_poi_artifacts(poi_metadata_path: Path, poi_texts_path: Path):
    poi_meta = load_json_list(poi_metadata_path)
    with open(poi_texts_path, "r", encoding="utf-8") as handle:
        poi_texts = json.load(handle)

    if not isinstance(poi_texts, list):
        raise ValueError(f"Expected a JSON list in {poi_texts_path}")
    if len(poi_meta) != len(poi_texts):
        raise ValueError(
            f"POI artifact sizes do not match: metadata={len(poi_meta)}, texts={len(poi_texts)}"
        )

    by_city: Dict[Tuple[str, str], List[int]] = {}
    for idx, row in enumerate(poi_meta):
        city = normalize_key(row.get("city", ""))
        country = normalize_key(row.get("country", ""))
        if city:
            by_city.setdefault((city, country), []).append(idx)

    return poi_meta, poi_texts, by_city


def load_cities_metadata(csv_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            city = str(row.get("city", "")).strip()
            country = str(row.get("country", "")).strip()
            description = str(row.get("description", "") or "").strip()
            tourist_value = max(0.0, min(1.0, parse_float(row.get("tourist_value_score", 0.0), 0.0)))
            tags_json = parse_tags_json(row.get("tags_json"))
            tag_text = str(row.get("tag_text", "") or "").strip()

            primary_name = city
            rows.append(
                {
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


def build_poi_lookup(poi_csv_path: Path) -> Dict[Tuple[str, str], List[str]]:
    poi_lookup: Dict[Tuple[str, str], List[str]] = {}

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


def get_city_poi_text(city_row: Dict[str, Any], poi_lookup: Dict[Tuple[str, str], List[str]], max_items: int = 25) -> str:
    city_key = normalize_key(city_row.get("name", ""))
    country_key = normalize_key(city_row.get("country", ""))

    pois = poi_lookup.get((city_key, country_key))
    if not pois:
        pois = poi_lookup.get((city_key, ""), [])

    if not pois:
        return ""

    joined = " ".join(pois[:max_items]).strip()
    return joined[:5000]


def cosine_distance(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 1.0
    cosine_sim = float(np.dot(a, b) / denom)
    cosine_sim = max(-1.0, min(1.0, cosine_sim))
    return 1.0 - ((cosine_sim + 1.0) / 2.0)


def normalize_distance(distance: float, min_distance: float, max_distance: float) -> float:
    span = max(max_distance - min_distance, 1e-12)
    d = (float(distance) - float(min_distance)) / span
    return max(0.0, min(1.0, d))


def distance_to_score(normalized_distance_value: float) -> float:
    d = max(0.0, min(1.0, float(normalized_distance_value)))
    return 1.0 - d


def calculate_description_score(faiss_distance: float, min_faiss_distance: float, max_faiss_distance: float) -> Tuple[float, float]:
    normalized = normalize_distance(faiss_distance, min_faiss_distance, max_faiss_distance)
    return distance_to_score(normalized), normalized


def calculate_poi_score(query_embedding: np.ndarray, poi_embedding: np.ndarray | None) -> Tuple[float, float]:
    if poi_embedding is None:
        return 0.0, 1.0
    poi_distance = cosine_distance(query_embedding, poi_embedding)
    poi_distance = max(0.0, min(1.0, poi_distance))
    return distance_to_score(poi_distance), poi_distance


def calculate_poi_score_from_hits(city_key: Tuple[str, str], city_best_poi_scores: Dict[Tuple[str, str], float]) -> Tuple[float, float]:
    score = max(0.0, min(1.0, float(city_best_poi_scores.get(city_key, 0.0))))
    if score <= 0.0:
        return 0.0, 1.0
    return score, 1.0 - score


def calculate_tags_score(query_tags_embedding: np.ndarray, city_tags_embedding: np.ndarray | None, tourist_value: float) -> Tuple[float, float]:
    if city_tags_embedding is None:
        return 0.0, 1.0
    tags_distance = cosine_distance(query_tags_embedding, city_tags_embedding)
    tags_distance = max(0.0, min(1.0, tags_distance))
    base_tags_score = distance_to_score(tags_distance)
    weighted_tags_score = base_tags_score * max(0.0, min(1.0, float(tourist_value)))
    return weighted_tags_score, tags_distance


def calculate_final_score(
    description_score: float,
    poi_score: float,
    tags_score: float,
    description_weight: float = 1.0,
    poi_weight: float = 1.0,
    tags_weight: float = 1.0,
) -> float:
    dw = max(0.0, float(description_weight))
    pw = max(0.0, float(poi_weight))
    tw = max(0.0, float(tags_weight))
    denom = dw + pw + tw
    if denom <= 1e-12:
        return (float(description_score) + float(poi_score) + float(tags_score)) / 3.0
    return (dw * float(description_score) + pw * float(poi_score) + tw * float(tags_score)) / denom


def build_city_tags_text(city_row: Dict[str, Any]) -> str:
    tags_json = city_row.get("tags_json") or []
    if tags_json:
        parts = [f"{item.get('tag', '')}:{parse_float(item.get('weight', 0.0), 0.0):.4f}" for item in tags_json]
        text = ", ".join([part for part in parts if part.strip(":")])
        if text:
            return text
    return str(city_row.get("tag_text", "") or "")


def search_candidates(query_vector: np.ndarray, embeddings: np.ndarray, candidate_k: int) -> Tuple[np.ndarray, np.ndarray]:
    deltas = embeddings - query_vector
    distances = np.sum(deltas * deltas, axis=1)
    indices = np.argsort(distances)[:candidate_k]
    return distances[indices], indices


class CityMatcher:
    def __init__(
        self,
        index_path: Path,
        embeddings_path: Path,
        city_metadata_path: Path,
        city_description_metadata_path: Path,
        deepseek_tags_csv: Path,
        filtered_cities_descriptions_csv: Path,
        wikivoyage_preprocessed_csv: Path,
        poi_index_path: Path,
        poi_metadata_path: Path,
        poi_texts_path: Path,
        embed_url: str,
        embed_model: str,
    ):
        self.index_path = index_path
        self.embeddings_path = embeddings_path
        self.city_metadata_path = city_metadata_path
        self.city_description_metadata_path = city_description_metadata_path
        self.deepseek_tags_csv = deepseek_tags_csv
        self.filtered_cities_descriptions_csv = filtered_cities_descriptions_csv
        self.wikivoyage_preprocessed_csv = wikivoyage_preprocessed_csv
        self.poi_index_path = poi_index_path
        self.poi_metadata_path = poi_metadata_path
        self.poi_texts_path = poi_texts_path
        self.embed_url = embed_url
        self.embed_model = embed_model

        self.metadata, self.city_embeddings = load_preembedded_city_metadata(
            city_metadata_path=city_metadata_path,
            city_description_metadata_path=city_description_metadata_path,
            deepseek_tags_csv=deepseek_tags_csv,
            filtered_cities_descriptions_csv=filtered_cities_descriptions_csv,
            wikivoyage_preprocessed_csv=wikivoyage_preprocessed_csv,
            embeddings_path=embeddings_path,
        )
        self.poi_metadata, self.poi_texts, self.poi_by_city = load_preembedded_poi_artifacts(
            poi_metadata_path=poi_metadata_path,
            poi_texts_path=poi_texts_path,
        )

        self.index = None
        self.embeddings = self.city_embeddings
        self.poi_index = None
        self.backend = "numpy"

        self._load_search_backend()
        self.usable_total = min(self._search_total(), len(self.metadata), int(self.city_embeddings.shape[0]))
        self.poi_total = len(self.poi_metadata)
        if self.usable_total <= 0:
            raise RuntimeError("No overlapping city entries were loaded.")

    def _load_search_backend(self) -> None:
        if self.index_path.exists():
            try:
                import faiss  # type: ignore

                self.index = faiss.read_index(str(self.index_path))
                self.backend = "faiss"
            except Exception:
                self.index = None

        if self.embeddings_path.exists():
            self.embeddings = np.load(self.embeddings_path).astype(np.float32)
        else:
            self.embeddings = self.city_embeddings

        if self.poi_index_path.exists():
            try:
                import faiss  # type: ignore

                self.poi_index = faiss.read_index(str(self.poi_index_path))
            except Exception:
                self.poi_index = None

        if self.index is None and self.embeddings is None:
            raise FileNotFoundError(
                f"Neither FAISS index nor embeddings file could be loaded: {self.index_path} / {self.embeddings_path}"
            )

    def _search_total(self) -> int:
        if self.index is not None:
            return int(self.index.ntotal)
        if self.embeddings is not None:
            return int(self.embeddings.shape[0])
        return 0

    def _search(self, query_vector: np.ndarray, candidate_k: int) -> Tuple[np.ndarray, np.ndarray]:
        candidate_k = max(1, min(int(candidate_k), self.usable_total))
        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)

        if self.index is not None:
            distances, indices = self.index.search(query, candidate_k)
            return distances[0], indices[0]

        if self.embeddings is None:
            raise RuntimeError("Search backend is unavailable.")

        distances, indices = search_candidates(query[0], self.embeddings[: self.usable_total], candidate_k)
        return distances, indices

    def _search_poi_hits(self, query_vector: np.ndarray, candidate_k: int) -> Dict[Tuple[str, str], float]:
        if self.poi_index is None:
            return {}

        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        candidate_k = max(1, min(int(candidate_k), self.poi_total))
        distances, indices = self.poi_index.search(query, candidate_k)

        best_scores: Dict[Tuple[str, str], float] = {}
        for poi_dist, poi_idx in zip(distances[0], indices[0]):
            poi_idx = int(poi_idx)
            if poi_idx < 0 or poi_idx >= self.poi_total:
                continue
            poi_row = self.poi_metadata[poi_idx]
            city_key = (normalize_key(poi_row.get("city", "")), normalize_key(poi_row.get("country", "")))
            chunk_score = 1.0 / (1.0 + max(0.0, float(poi_dist)))
            best_scores[city_key] = max(best_scores.get(city_key, 0.0), chunk_score)

        return best_scores

    def rank(
        self,
        tags: Sequence[Dict[str, float]],
        top_k: int = 10,
        candidate_k: int = 120,
        description_weight: float = 1.0,
        poi_weight: float = 1.0,
        tags_weight: float = 1.0,
    ) -> List[Dict[str, Any]]:
        normalized_tags = normalize_input_tags(tags)
        if not normalized_tags:
            raise ValueError("Payload must contain at least one tag object with a positive weight.")

        query_embedding = embed_weighted_tags(normalized_tags, self.embed_url, self.embed_model)
        query_tags_embedding = embed_texts([build_tags_text(normalized_tags)], self.embed_url, self.embed_model)[0]

        candidate_k = min(max(1, int(candidate_k)), self.usable_total)
        distances, indices = self._search(query_embedding, candidate_k)
        poi_best_scores = self._search_poi_hits(query_embedding, max(200, candidate_k * 4))

        candidate_rows: List[Tuple[int, float, Dict[str, Any]]] = []
        valid_faiss_distances: List[float] = []
        for faiss_dist, idx in zip(distances, indices):
            idx = int(idx)
            if idx < 0 or idx >= self.usable_total:
                continue
            row = self.metadata[idx]
            valid_faiss_distances.append(float(faiss_dist))
            candidate_rows.append((idx, float(faiss_dist), row))

        if not candidate_rows:
            return []

        min_faiss_dist = min(valid_faiss_distances)
        max_faiss_dist = max(valid_faiss_distances)

        ranked: List[Dict[str, Any]] = []
        for idx, faiss_dist, row in candidate_rows:
            description_score, description_distance_norm = calculate_description_score(
                faiss_distance=faiss_dist,
                min_faiss_distance=min_faiss_dist,
                max_faiss_distance=max_faiss_dist,
            )

            city_key = (normalize_key(row.get("city", "") or row.get("name", "")), normalize_key(row.get("country", "")))
            poi_score, poi_distance = calculate_poi_score_from_hits(city_key, poi_best_scores)
            tourist_value = parse_float(row.get("tourist_value", 0.0), 0.0)
            tags_score, tags_distance = calculate_tags_score(
                query_tags_embedding=query_tags_embedding,
                city_tags_embedding=self.city_embeddings[idx],
                tourist_value=tourist_value,
            )

            final_score = calculate_final_score(
                description_score=description_score,
                poi_score=poi_score,
                tags_score=tags_score,
                description_weight=description_weight,
                poi_weight=poi_weight,
                tags_weight=tags_weight,
            )

            ranked.append(
                {
                    "idx": idx,
                    "city": row.get("city", "") or row.get("name", ""),
                    "name": row.get("city", "") or row.get("name", ""),
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

        ranked.sort(key=lambda item: item["final_score"], reverse=True)
        return ranked[: max(1, int(top_k))]


def _serialize_request_payload(payload: Any) -> Tuple[List[Dict[str, float]], Dict[str, Any]]:
    if isinstance(payload, list):
        tags = normalize_input_tags(payload)
        options: Dict[str, Any] = {}
        return tags, options

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON array or object.")

    raw_tags = payload.get("tags", payload.get("input", payload.get("items", [])))
    tags = normalize_input_tags(raw_tags)
    options = {
        "top_k": payload.get("top_k", 10),
        "final_k": payload.get("final_k", None),
        "candidate_k": payload.get("candidate_k", 120),
        "description_weight": payload.get("description_weight", 1.0),
        "poi_weight": payload.get("poi_weight", 1.0),
        "tags_weight": payload.get("tags_weight", 1.0),
    }
    return tags, options


def _build_matcher() -> CityMatcher:
    return CityMatcher(
        index_path=Path(os.getenv("CITY_INDEX_PATH", str(DEFAULT_INDEX_PATH))),
        embeddings_path=Path(os.getenv("CITY_EMBEDDINGS_PATH", str(DEFAULT_EMBEDDINGS_PATH))),
        city_metadata_path=Path(os.getenv("CITY_METADATA_PATH", str(DEFAULT_CITY_METADATA_PATH))),
        city_description_metadata_path=Path(
            os.getenv("CITY_DESCRIPTION_METADATA_PATH", str(DEFAULT_CITY_DESCRIPTION_METADATA_PATH))
        ),
        deepseek_tags_csv=Path(os.getenv("DEEPSEEK_TAGS_CSV_PATH", str(DEFAULT_DEEPSEEK_TAGS_CSV))),
        filtered_cities_descriptions_csv=Path(
            os.getenv("FILTERED_CITIES_DESCRIPTIONS_CSV_PATH", str(DEFAULT_FILTERED_CITIES_DESCRIPTIONS_CSV))
        ),
        wikivoyage_preprocessed_csv=Path(
            os.getenv("WIKIVOYAGE_PREPROCESSED_CSV_PATH", str(DEFAULT_WIKIVOYAGE_PREPROCESSED_CSV))
        ),
        poi_index_path=Path(os.getenv("POI_INDEX_PATH", str(DEFAULT_POI_INDEX_PATH))),
        poi_metadata_path=Path(os.getenv("POI_METADATA_PATH", str(DEFAULT_POI_METADATA_PATH))),
        poi_texts_path=Path(os.getenv("POI_TEXTS_PATH", str(DEFAULT_POI_TEXTS_PATH))),
        embed_url=EMBED_URL,
        embed_model=EMBED_MODEL,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global matcher
    matcher = _build_matcher()
    yield


app = FastAPI(title="City Serving API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

matcher: CityMatcher | None = None
_JOB_LOCK = threading.Lock()
_JOB_CACHE: Dict[str, Dict[str, Any]] = {}


def _ensure_jobs_dir() -> None:
    BOOKING_JOBS_DIR.mkdir(parents=True, exist_ok=True)


def _job_path(job_id: str) -> Path:
    return BOOKING_JOBS_DIR / f"{job_id}.json"


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _save_job_record(job: Dict[str, Any]) -> None:
    _ensure_jobs_dir()
    job["updated_at"] = _now_iso()
    job_path = _job_path(str(job["job_id"]))
    tmp_path = job_path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(job, handle, ensure_ascii=False, indent=2)
    tmp_path.replace(job_path)
    with _JOB_LOCK:
        _JOB_CACHE[str(job["job_id"])] = job


def _load_job_record(job_id: str) -> Dict[str, Any] | None:
    with _JOB_LOCK:
        cached = _JOB_CACHE.get(job_id)
    if cached is not None:
        return cached

    job_path = _job_path(job_id)
    if not job_path.exists():
        return None
    with job_path.open("r", encoding="utf-8") as handle:
        job = json.load(handle)
    if isinstance(job, dict):
        with _JOB_LOCK:
            _JOB_CACHE[job_id] = job
        return job
    return None


def _normalize_batch_requests(payload: Any) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
    searches = payload.get("requests", payload.get("searches", payload.get("items", [])))
    if not isinstance(searches, list) or not searches:
        raise HTTPException(status_code=400, detail="Provide a non-empty 'requests' array.")
    normalized: List[Dict[str, Any]] = []
    for item in searches:
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail="Each request must be an object.")
        request_type = str(item.get("type", "")).strip().lower()
        if request_type not in {"hotels", "attractions", "flights", "weather"}:
            raise HTTPException(status_code=400, detail=f"Unsupported request type: {request_type!r}")
        normalized.append(item)
    return normalized


def _run_weather_search(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from booking.open_meteo_info import geocode_city, get_weather
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Open-Meteo module import failed: {exc}") from exc

    city = str(payload.get("city", "")).strip()
    country = str(payload.get("country", "")).strip() or None
    lat = payload.get("lat")
    lon = payload.get("lon")
    days = int(payload.get("days", 3))

    if lat is not None and lon is not None:
        place = {
            "name": city or "Custom location",
            "country": country or "",
            "latitude": float(lat),
            "longitude": float(lon),
        }
    else:
        if not city:
            raise HTTPException(status_code=400, detail="Provide city or both lat/lon.")
        place = geocode_city(city, country)

    weather = get_weather(float(place["latitude"]), float(place["longitude"]), days)
    return {
        "location": {
            "name": place.get("name", ""),
            "country": place.get("country", ""),
            "latitude": place.get("latitude"),
            "longitude": place.get("longitude"),
        },
        "weather": weather,
    }


def _call_scraper_method(scraper: Any, method_names: List[str], *args: Any, **kwargs: Any) -> Any:
    for method_name in method_names:
        method = getattr(scraper, method_name, None)
        if callable(method):
            return method(*args, **kwargs)
    raise AttributeError(f"{scraper.__class__.__name__} does not provide any of {method_names}")


def _load_scraper_cookies(scraper: Any, cookies_file: str) -> bool:
    if not cookies_file:
        return False
    for method_name in ("load_cookies", "_load_cookies"):
        method = getattr(scraper, method_name, None)
        if callable(method):
            try:
                return bool(method(cookies_file))
            except Exception:
                return False
    return False


def _normalize_browser_hotels(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        rating_raw = row.get("review_score")
        rating_value = None
        if rating_raw not in (None, "", "N/A"):
            try:
                rating_value = parse_float(rating_raw, 0.0)
            except Exception:
                rating_value = None
        normalized.append(
            {
                "name": row.get("title", "N/A"),
                "url": row.get("url", "N/A"),
                "location": row.get("location", "N/A"),
                "description": row.get("description", "N/A"),
                "rating": rating_value,
                "rating_label": row.get("review_label", "N/A"),
                "review_count": row.get("review_count", "N/A"),
                "price_per_night": row.get("price", "N/A"),
                "currency": "N/A",
                "availability": row.get("availability", "N/A"),
                "image_alt": "N/A",
                "image_url": "N/A",
            }
        )
    return normalized


def _run_hotels_search(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from booking.booking_hotels import (
            _cookie_header,
            _load_cookie_records,
            _parse_proxy_line,
            build_search_url,
            default_dates,
            extract_hotels,
            fetch_search_html,
            load_proxies,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Hotels module import failed: {exc}") from exc

    city = str(payload.get("city", "")).strip()
    country = str(payload.get("country", "")).strip()
    if not city or not country:
        raise HTTPException(status_code=400, detail="Both city and country are required.")

    checkin = payload.get("checkin")
    checkout = payload.get("checkout")
    days = int(payload.get("days", 1))
    adults = int(payload.get("adults", 2))
    rooms = int(payload.get("rooms", 1))
    limit = int(payload.get("limit", 25))
    min_rating = payload.get("min_rating")
    max_price = payload.get("max_price")
    proxy_file = payload.get("proxy_file")
    proxy = payload.get("proxy")
    no_proxy = bool(payload.get("no_proxy", False))
    cookies_file = payload.get("cookies_file")
    save_cookies = payload.get("save_cookies")
    manual_unblock = bool(payload.get("manual_unblock", False))
    debug_browser = bool(payload.get("debug_browser", False))
    headless = bool(payload.get("headless", False))
    page_load_timeout = int(payload.get("page_load_timeout", 45))
    content_timeout = int(payload.get("content_timeout", 35))
    profile_dir = payload.get("profile_dir")
    profile_name = str(payload.get("profile_name", "Default"))
    use_browser = bool(payload.get("use_browser", True))

    if use_browser:
        try:
            from booking.booking_hotels import build_search_url, default_dates
            from booking.booking_hotels_scraper import BookingHotelsScraper
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Hotels Selenium module import failed: {exc}") from exc

        if checkin and not checkout:
            try:
                checkout_dt = datetime.strptime(str(checkin), "%Y-%m-%d")
                checkout = (checkout_dt + timedelta(days=max(1, days))).date().isoformat()
            except Exception as exc:
                raise HTTPException(status_code=400, detail="checkin must be YYYY-MM-DD when checkout is omitted.") from exc
        elif not checkin and not checkout:
            checkin, checkout = default_dates()
        elif not (checkin and checkout):
            raise HTTPException(status_code=400, detail="Provide both checkin and checkout, or neither.")

        scraper = BookingHotelsScraper(headless=headless)
        try:
            _load_scraper_cookies(scraper, cookies_file or "")
            if not scraper.search_hotels(
                city=city,
                country=country,
                checkin=str(checkin),
                checkout=str(checkout),
                adults=adults,
                children=int(payload.get("children", 0)),
                rooms=rooms,
                pets=bool(payload.get("pets", False)),
            ):
                raise HTTPException(status_code=502, detail="Booking hotels search failed.")
            hotel_rows = scraper.scrape_hotels()[:limit]
            return {
                "query": {
                    "city": city,
                    "country": country,
                    "checkin": str(checkin),
                    "checkout": str(checkout),
                    "adults": adults,
                    "rooms": rooms,
                    "constraints": {
                        "min_rating": min_rating,
                        "max_price": max_price,
                    },
                },
                "search_url": build_search_url(
                    city=city,
                    country=country,
                    checkin=str(checkin),
                    checkout=str(checkout),
                    adults=adults,
                    rooms=rooms,
                ),
                "total_found": len(hotel_rows),
                "hotels": _normalize_browser_hotels(hotel_rows),
            }
        finally:
            scraper.close()

    cookie_records = _load_cookie_records(cookies_file or "")
    cookie_header = _cookie_header(cookie_records)
    has_session_state = bool(profile_dir or cookie_records)

    if no_proxy:
        proxy_urls: List[str] = []
    elif proxy:
        proxy_value = _parse_proxy_line(str(proxy))
        if not proxy_value:
            raise HTTPException(status_code=400, detail="Invalid proxy format. Use host:port or host:port:user:pass")
        proxy_urls = [proxy_value]
    else:
        proxy_urls = load_proxies(str(proxy_file)) if proxy_file else []
        if has_session_state:
            proxy_urls = []

    if checkin and not checkout:
        try:
            checkout_dt = datetime.strptime(str(checkin), "%Y-%m-%d")
            checkout = (checkout_dt + timedelta(days=max(1, days))).date().isoformat()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="checkin must be YYYY-MM-DD when checkout is omitted.") from exc
    elif not checkin and not checkout:
        checkin, checkout = default_dates()
    elif not (checkin and checkout):
        raise HTTPException(status_code=400, detail="Provide both checkin and checkout, or neither.")

    search_url = build_search_url(
        city=city,
        country=country,
        checkin=str(checkin),
        checkout=str(checkout),
        adults=adults,
        rooms=rooms,
    )

    html_text = fetch_search_html(
        search_url,
        proxy_urls=proxy_urls,
        cookie_header=cookie_header,
        cookie_records=cookie_records,
        profile_dir=profile_dir,
        profile_name=profile_name,
        save_cookies_file=save_cookies,
        prefer_browser=has_session_state,
        manual_unblock=manual_unblock,
        debug_browser=debug_browser,
        headless=headless,
        page_load_timeout=page_load_timeout,
        content_timeout=content_timeout,
    )
    hotels = extract_hotels(
        html_text,
        city=city,
        country=country,
        limit=limit,
        min_rating=parse_float(min_rating, 0.0) if min_rating is not None else None,
        max_price=parse_float(max_price, 0.0) if max_price is not None else None,
    )

    return {
        "query": {
            "city": city,
            "country": country,
            "checkin": str(checkin),
            "checkout": str(checkout),
            "adults": adults,
            "rooms": rooms,
            "constraints": {
                "min_rating": min_rating,
                "max_price": max_price,
            },
        },
        "search_url": search_url,
        "total_found": len(hotels),
        "hotels": hotels,
    }


def _run_attractions_search(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from booking.booking_attractions_scraper import BookingAttractionsScraper
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Attractions module import failed: {exc}") from exc

    city = str(payload.get("city", "")).strip()
    country = str(payload.get("country", "")).strip()
    if not city or not country:
        raise HTTPException(status_code=400, detail="Both city and country are required.")

    date_value = payload.get("date")
    output = payload.get("output")
    cookies_file = payload.get("cookies_file", "booking_cookies.json")
    headless = bool(payload.get("headless", False))

    scraper = BookingAttractionsScraper(headless=headless)
    try:
        _load_scraper_cookies(scraper, cookies_file)
        if not _call_scraper_method(scraper, ["search_attractions", "search"], city, country, date_value):
            raise HTTPException(status_code=502, detail="Booking attractions search failed.")
        attractions = _call_scraper_method(scraper, ["scrape_attractions", "scrape"])
        if output:
            _call_scraper_method(scraper, ["save_to_csv", "save_csv"], attractions, output)
        return {
            "query": {
                "city": city,
                "country": country,
                "date": date_value,
            },
            "total_found": len(attractions),
            "attractions": attractions,
        }
    finally:
        scraper.close()


def _run_flights_search(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from booking.flight_search import search_flights
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Flight search module import failed: {exc}") from exc

    from_city = str(payload.get("from_city", "")).strip()
    to_city = str(payload.get("to_city", "")).strip()
    date = str(payload.get("date", "")).strip()
    if not from_city or not to_city or not date:
        raise HTTPException(status_code=400, detail="from_city, to_city, and date are required.")

    serpapi_key = str(payload.get("serpapi_key", "") or os.getenv("SERPAPI_KEY", "")).strip()
    if not serpapi_key:
        raise HTTPException(status_code=400, detail="SerpApi key is required via serpapi_key or SERPAPI_KEY.")

    return search_flights(
        from_city=from_city,
        to_city=to_city,
        date=date,
        serpapi_key=serpapi_key,
        from_country=payload.get("from_country"),
        to_country=payload.get("to_country"),
        from_iata=payload.get("from_iata"),
        to_iata=payload.get("to_iata"),
        adults=int(payload.get("adults", 1)),
        cabin=str(payload.get("cabin", "economy")),
        max_stops=payload.get("max_stops"),
        currency=str(payload.get("currency", "USD")),
        return_date=payload.get("return_date"),
    )


def _run_booking_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    request_type = str(payload.get("type", "")).strip().lower()
    if request_type == "weather":
        return _run_weather_search(payload)
    if request_type == "hotels":
        return _run_hotels_search(payload)
    if request_type == "attractions":
        return _run_attractions_search(payload)
    if request_type == "flights":
        return _run_flights_search(payload)
    raise HTTPException(status_code=400, detail=f"Unsupported request type: {request_type!r}")


def _process_booking_search_job(job_id: str) -> None:
    logger.info(f"[JOB {job_id}] Worker thread starting")
    job = _load_job_record(job_id)
    if job is None:
        logger.error(f"[JOB {job_id}] Job record not found")
        return

    try:
        job["status"] = "running"
        job["started_at"] = _now_iso()
        job.setdefault("results", [])
        _save_job_record(job)
        logger.info(f"[JOB {job_id}] Job status set to running, {len(job.get('requests', []))} requests")

        results: List[Dict[str, Any]] = []
        for index, request_payload in enumerate(job.get("requests", [])):
            logger.info(f"[JOB {job_id}] Processing request {index + 1}/{len(job.get('requests', []))} - type: {request_payload.get('type')}")
            result = _run_booking_request(request_payload)
            entry = {
                "index": index,
                "type": request_payload.get("type"),
                "request": request_payload,
                "result": result,
            }
            results.append(entry)
            job["results"] = results
            job["progress"] = {
                "completed": index + 1,
                "total": len(job.get("requests", [])),
            }
            _save_job_record(job)
            logger.info(f"[JOB {job_id}] Request {index + 1} completed")

        job["status"] = "done"
        job["finished_at"] = _now_iso()
        job["results"] = results
        _save_job_record(job)
        logger.info(f"[JOB {job_id}] Job completed successfully")
    except HTTPException as exc:
        logger.error(f"[JOB {job_id}] HTTPException: {exc.status_code} - {exc.detail}")
        job["status"] = "error"
        job["finished_at"] = _now_iso()
        job["error"] = {"status_code": exc.status_code, "detail": exc.detail}
        _save_job_record(job)
    except Exception as exc:
        logger.error(f"[JOB {job_id}] Exception: {exc}", exc_info=True)
        job["status"] = "error"
        job["finished_at"] = _now_iso()
        job["error"] = str(exc)
        _save_job_record(job)




@app.get("/")
def root() -> Dict[str, Any]:
    if matcher is None:
        raise HTTPException(status_code=503, detail="Service is still starting.")
    return {
        "service": "city-serving-api",
        "backend": matcher.backend,
        "cities": matcher.usable_total,
        "city_index_path": str(matcher.index_path),
        "city_metadata_path": str(matcher.city_metadata_path),
        "poi_index_path": str(matcher.poi_index_path),
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    if matcher is None:
        raise HTTPException(status_code=503, detail="Service is still starting.")
    return {
        "status": "ok",
        "backend": matcher.backend,
        "cities": matcher.usable_total,
    }


@app.post("/nearest-city")
async def nearest_city(request: Request) -> Dict[str, Any]:
    if matcher is None:
        raise HTTPException(status_code=503, detail="Service is still starting.")

    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.") from exc

    tags, options = _serialize_request_payload(payload)
    try:
        results = matcher.rank(
            tags=tags,
            top_k=int(options.get("top_k", 10)),
            candidate_k=int(options.get("candidate_k", 120)),
            description_weight=parse_float(options.get("description_weight", 1.0), 1.0),
            poi_weight=parse_float(options.get("poi_weight", 1.0), 1.0),
            tags_weight=parse_float(options.get("tags_weight", 1.0), 1.0),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Embedding service request failed: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Apply final_k slicing if provided
    final_k = options.get("final_k")
    if final_k is not None:
        try:
            final_k = int(final_k)
            results = results[:final_k]
        except (TypeError, ValueError):
            pass

    return {
        "input": tags,
        "query_text": build_query_text(tags),
        "tags_text": build_tags_text(tags),
        "backend": matcher.backend,
        "best_match": results[0] if results else None,
        "results": results,
    }


@app.post("/booking/weather")
async def booking_weather(request: Request) -> Dict[str, Any]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.") from exc

    city = str(payload.get("city", "")).strip()
    country = str(payload.get("country", "")).strip() or None
    lat = payload.get("lat")
    lon = payload.get("lon")
    days = int(payload.get("days", 3))
    as_json = bool(payload.get("json", True))

    try:
        from booking.open_meteo_info import geocode_city, get_weather
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Open-Meteo module import failed: {exc}") from exc

    try:
        if lat is not None and lon is not None:
            place = {
                "name": city or "Custom location",
                "country": country or "",
                "latitude": float(lat),
                "longitude": float(lon),
            }
        else:
            if not city:
                raise HTTPException(status_code=400, detail="Provide city or both lat/lon.")
            place = geocode_city(city, country)

        weather = get_weather(float(place["latitude"]), float(place["longitude"]), days)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if not as_json:
        return {
            "location": {
                "name": place.get("name", ""),
                "country": place.get("country", ""),
                "latitude": place.get("latitude"),
                "longitude": place.get("longitude"),
            },
            "weather": weather,
        }

    return {
        "requested_at": datetime.utcnow().isoformat() + "Z",
        "location": {
            "name": place.get("name", ""),
            "country": place.get("country", ""),
            "latitude": place.get("latitude"),
            "longitude": place.get("longitude"),
        },
        "weather": weather,
    }


@app.post("/booking/hotels")
async def booking_hotels(request: Request) -> Dict[str, Any]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.") from exc

    city = str(payload.get("city", "")).strip()
    country = str(payload.get("country", "")).strip()
    if not city or not country:
        raise HTTPException(status_code=400, detail="Both city and country are required.")

    checkin = payload.get("checkin")
    checkout = payload.get("checkout")
    days = int(payload.get("days", 1))
    adults = int(payload.get("adults", 2))
    rooms = int(payload.get("rooms", 1))
    limit = int(payload.get("limit", 25))
    min_rating = payload.get("min_rating")
    max_price = payload.get("max_price")
    proxy_file = payload.get("proxy_file")
    proxy = payload.get("proxy")
    no_proxy = bool(payload.get("no_proxy", False))
    cookies_file = payload.get("cookies_file")
    save_cookies = payload.get("save_cookies")
    manual_unblock = bool(payload.get("manual_unblock", False))
    debug_browser = bool(payload.get("debug_browser", False))
    headless = bool(payload.get("headless", False))
    page_load_timeout = int(payload.get("page_load_timeout", 45))
    content_timeout = int(payload.get("content_timeout", 35))
    profile_dir = payload.get("profile_dir")
    profile_name = str(payload.get("profile_name", "Default"))

    try:
        from booking.booking_hotels import (
            _cookie_header,
            _load_cookie_records,
            _parse_proxy_line,
            build_search_url,
            default_dates,
            extract_hotels,
            fetch_search_html,
            load_proxies,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Hotels module import failed: {exc}") from exc

    cookie_records = _load_cookie_records(cookies_file or "")
    cookie_header = _cookie_header(cookie_records)
    has_session_state = bool(profile_dir or cookie_records)

    if no_proxy:
        proxy_urls: List[str] = []
    elif proxy:
        proxy_value = _parse_proxy_line(str(proxy))
        if not proxy_value:
            raise HTTPException(status_code=400, detail="Invalid proxy format. Use host:port or host:port:user:pass")
        proxy_urls = [proxy_value]
    else:
        proxy_urls = load_proxies(str(proxy_file)) if proxy_file else []
        if has_session_state:
            proxy_urls = []

    if checkin and not checkout:
        try:
            checkout_dt = datetime.strptime(str(checkin), "%Y-%m-%d")
            checkout = (checkout_dt + timedelta(days=max(1, days))).date().isoformat()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="checkin must be YYYY-MM-DD when checkout is omitted.") from exc
    elif not checkin and not checkout:
        checkin, checkout = default_dates()
    elif not (checkin and checkout):
        raise HTTPException(status_code=400, detail="Provide both checkin and checkout, or neither.")

    search_url = build_search_url(
        city=city,
        country=country,
        checkin=str(checkin),
        checkout=str(checkout),
        adults=adults,
        rooms=rooms,
    )

    try:
        html_text = fetch_search_html(
            search_url,
            proxy_urls=proxy_urls,
            cookie_header=cookie_header,
            cookie_records=cookie_records,
            profile_dir=profile_dir,
            profile_name=profile_name,
            save_cookies_file=save_cookies,
            prefer_browser=has_session_state,
            manual_unblock=manual_unblock,
            debug_browser=debug_browser,
            headless=headless,
            page_load_timeout=page_load_timeout,
            content_timeout=content_timeout,
        )
        hotels = extract_hotels(
            html_text,
            city=city,
            country=country,
            limit=limit,
            min_rating=parse_float(min_rating, 0.0) if min_rating is not None else None,
            max_price=parse_float(max_price, 0.0) if max_price is not None else None,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {
        "query": {
            "city": city,
            "country": country,
            "checkin": str(checkin),
            "checkout": str(checkout),
            "adults": adults,
            "rooms": rooms,
            "constraints": {
                "min_rating": min_rating,
                "max_price": max_price,
            },
        },
        "search_url": search_url,
        "total_found": len(hotels),
        "hotels": hotels,
    }


@app.post("/booking/attractions")
async def booking_attractions(request: Request) -> Dict[str, Any]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.") from exc

    city = str(payload.get("city", "")).strip()
    country = str(payload.get("country", "")).strip()
    if not city or not country:
        raise HTTPException(status_code=400, detail="Both city and country are required.")

    date_value = payload.get("date")
    output = payload.get("output")
    cookies_file = payload.get("cookies_file", "booking_cookies.json")
    headless = bool(payload.get("headless", False))

    try:
        from booking.booking_attractions_scraper import BookingAttractionsScraper
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Attractions module import failed: {exc}") from exc

    scraper = BookingAttractionsScraper(headless=headless)
    try:
        _load_scraper_cookies(scraper, cookies_file)
        if not _call_scraper_method(scraper, ["search_attractions", "search"], city, country, date_value):
            raise HTTPException(status_code=502, detail="Booking attractions search failed.")
        attractions = _call_scraper_method(scraper, ["scrape_attractions", "scrape"])
        if output:
            _call_scraper_method(scraper, ["save_to_csv", "save_csv"], attractions, output)
        return {
            "query": {
                "city": city,
                "country": country,
                "date": date_value,
            },
            "total_found": len(attractions),
            "attractions": attractions,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    finally:
        scraper.close()


@app.post("/booking/flights")
async def booking_flights(request: Request) -> Dict[str, Any]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.") from exc

    from_city = str(payload.get("from_city", "")).strip()
    to_city = str(payload.get("to_city", "")).strip()
    date = str(payload.get("date", "")).strip()
    if not from_city or not to_city or not date:
        raise HTTPException(status_code=400, detail="from_city, to_city, and date are required.")

    serpapi_key = str(payload.get("serpapi_key", "") or os.getenv("SERPAPI_KEY", "")).strip()
    if not serpapi_key:
        raise HTTPException(status_code=400, detail="SerpApi key is required via serpapi_key or SERPAPI_KEY.")

    try:
        from booking.flight_search import search_flights
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Flight search module import failed: {exc}") from exc

    try:
        results = search_flights(
            from_city=from_city,
            to_city=to_city,
            date=date,
            serpapi_key=serpapi_key,
            from_country=payload.get("from_country"),
            to_country=payload.get("to_country"),
            from_iata=payload.get("from_iata"),
            to_iata=payload.get("to_iata"),
            adults=int(payload.get("adults", 1)),
            cabin=str(payload.get("cabin", "economy")),
            max_stops=payload.get("max_stops"),
            currency=str(payload.get("currency", "USD")),
            return_date=payload.get("return_date"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Flight search request failed: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return results


@app.post("/booking/search")
async def booking_search(request: Request) -> Dict[str, Any]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.") from exc

    searches = _normalize_batch_requests(payload)
    job_id = uuid.uuid4().hex
    job = {
        "job_id": job_id,
        "status": "running",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "requests": searches,
        "progress": {"completed": 0, "total": len(searches)},
        "results": [],
        "error": None,
    }
    _save_job_record(job)

    logger.info(f"Starting worker thread for job {job_id}")
    worker = threading.Thread(target=_process_booking_search_job, args=(job_id,), daemon=False)
    worker.start()
    time.sleep(0.1)  # Give thread a moment to start

    return {
        "job_id": job_id,
        "status": "running",
        "requests": len(searches),
    }


@app.get("/booking/search/{job_id}")
def booking_search_status(job_id: str) -> Dict[str, Any]:
    job = _load_job_record(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job




if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )


from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from serving.main import (
    DEFAULT_CITY_DESCRIPTION_METADATA_PATH,
    DEFAULT_CITY_METADATA_PATH,
    DEFAULT_EMBEDDINGS_PATH,
    DEFAULT_INDEX_PATH,
    DEFAULT_POI_INDEX_PATH,
    DEFAULT_POI_METADATA_PATH,
    DEFAULT_POI_TEXTS_PATH,
    DEFAULT_TOURIST_VALUE_CSV,
    EMBED_MODEL,
    EMBED_URL,
    CityMatcher,
)


def parse_json_payload() -> dict:
    """Parse JSON payload from stdin."""
    payload_text = sys.stdin.read()
    payload = json.loads(payload_text)
    return payload


def extract_tags_from_payload(payload: dict) -> list[dict[str, float]]:
    """Extract and filter tags from payload, removing zero weights."""
    tags_data = payload.get("tags", [])
    tags: list[dict[str, float]] = []
    
    for tag_obj in tags_data:
        if not isinstance(tag_obj, dict):
            continue
        tag_name = tag_obj.get("tag", "").strip()
        weight = float(tag_obj.get("weight", 0.0))
        
        # Skip tags with zero weight
        if weight == 0.0:
            continue
        
        if tag_name:
            tags.append({tag_name: weight})
    
    return tags


def main() -> None:
    # Parse JSON payload from stdin
    payload = parse_json_payload()
    
    # Extract parameters
    tags = extract_tags_from_payload(payload)
    top_k = payload.get("top_k", 30)
    final_k = payload.get("final_k", 20)
    
    if not tags:
        print("Error: No valid tags provided in payload.", file=sys.stderr)
        sys.exit(1)

    matcher = CityMatcher(
        index_path=Path(DEFAULT_INDEX_PATH),
        embeddings_path=Path(DEFAULT_EMBEDDINGS_PATH),
        city_metadata_path=Path(DEFAULT_CITY_METADATA_PATH),
        city_description_metadata_path=Path(DEFAULT_CITY_DESCRIPTION_METADATA_PATH),
        tourist_value_csv=Path(DEFAULT_TOURIST_VALUE_CSV),
        poi_index_path=Path(DEFAULT_POI_INDEX_PATH),
        poi_metadata_path=Path(DEFAULT_POI_METADATA_PATH),
        poi_texts_path=Path(DEFAULT_POI_TEXTS_PATH),
        embed_url=EMBED_URL,
        embed_model=EMBED_MODEL,
    )

    # Rank with top_k, then slice to final_k results
    results = matcher.rank(
        tags=tags,
        top_k=top_k,
        candidate_k=120,
        description_weight=1.0,
        poi_weight=1.0,
        tags_weight=1.0,
    )
    
    # Slice to final_k results
    final_results = results[:final_k]

    for rank, row in enumerate(final_results, start=1):
        print(
            f"{rank:02d}. {row['city']}, {row['country']} | "
            f"final={row['final_score']:.6f} | "
            f"desc={row['description_score']:.6f} | "
            f"poi={row['poi_score']:.6f} | "
            f"tags={row['tags_score']:.6f}"
        )


if __name__ == "__main__":
    main()

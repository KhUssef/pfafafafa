import os
import time
import json
import math
import requests
import numpy as np
import faiss

# =========================
# CONFIG
# =========================

OPENTRIPMAP_API_BASE = "https://api.opentripmap.com/0.1/en/places"
OPENTRIPMAP_API_KEY = "5ae2e3f221c38a28845f05b6ceb3a9aadb7373567e7da100af4723c6"

LM_STUDIO_URL = "http://127.0.0.1:1234/v1/embeddings"
EMBED_MODEL = "qwen/text-embedding-qwen3-embedding-0.6b"

# OpenTripMap paging for each tile
LIMIT = 500

# World scan settings. Smaller tiles = more complete but slower.
MIN_LAT = -85.0
MAX_LAT = 85.0
MIN_LON = -180.0
MAX_LON = 180.0
TILE_SIZE_DEGREES = 2.0

# Filter to tourist attractions / interesting places
KINDS_FILTER = "interesting_places"

# Reliability and pacing
REQUEST_TIMEOUT = (10, 120)
MAX_HTTP_RETRIES = 5
RETRY_BACKOFF_SECONDS = 2
SLEEP_BETWEEN_CALLS = 0.2

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Chunking
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120

# Test mode target API calls
TEST_REQUESTS = 100

# =========================
# STORAGE
# =========================

index = None
texts = []
metadata = []


# =========================
# HTTP HELPERS
# =========================

class RequestBudgetExceeded(Exception):
    pass


class RequestTracker:
    def __init__(self, max_requests=None):
        self.max_requests = max_requests
        self.calls = 0

    def consume(self):
        if self.max_requests is not None and self.calls >= self.max_requests:
            raise RequestBudgetExceeded(
                f"Request budget reached: {self.calls}/{self.max_requests}"
            )
        self.calls += 1


def get_json_with_retry(url, *, params=None, headers=None, timeout=REQUEST_TIMEOUT, request_name="request", tracker=None):
    last_error = None

    for attempt in range(1, MAX_HTTP_RETRIES + 1):
        try:
            if tracker is not None:
                tracker.consume()

            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout
            )

            if response.status_code in RETRYABLE_STATUS_CODES:
                raise requests.exceptions.HTTPError(
                    f"Retryable HTTP status: {response.status_code}",
                    response=response
                )

            response.raise_for_status()
            return response.json()

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.HTTPError) as exc:
            last_error = exc

            is_retryable_http = (
                isinstance(exc, requests.exceptions.HTTPError)
                and exc.response is not None
                and exc.response.status_code in RETRYABLE_STATUS_CODES
            )

            is_retryable = (
                isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError))
                or is_retryable_http
            )

            if not is_retryable or attempt == MAX_HTTP_RETRIES:
                raise

            backoff = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
            print(
                f"{request_name} failed (attempt {attempt}/{MAX_HTTP_RETRIES}): {exc}. "
                f"Retrying in {backoff}s..."
            )
            time.sleep(backoff)

    raise RuntimeError(f"{request_name} failed after retries: {last_error}")


# =========================
# OPEN TRIP MAP FETCH
# =========================

def validate_api_key():
    if not OPENTRIPMAP_API_KEY:
        raise RuntimeError(
            "Missing OpenTripMap API key. Set OPENTRIPMAP_API_KEY in your environment."
        )


def iter_tiles(tile_size_deg=TILE_SIZE_DEGREES):
    lat = MIN_LAT
    while lat < MAX_LAT:
        lat_next = min(lat + tile_size_deg, MAX_LAT)

        lon = MIN_LON
        while lon < MAX_LON:
            lon_next = min(lon + tile_size_deg, MAX_LON)
            yield (lon, lon_next, lat, lat_next)
            lon = lon_next

        lat = lat_next


def fetch_bbox_page(lon_min, lon_max, lat_min, lat_max, offset=0, tracker=None):
    params = {
        "lon_min": lon_min,
        "lon_max": lon_max,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "kinds": KINDS_FILTER,
        "format": "json",
        "limit": LIMIT,
        "offset": offset,
        "apikey": OPENTRIPMAP_API_KEY,
    }

    data = get_json_with_retry(
        f"{OPENTRIPMAP_API_BASE}/bbox",
        params=params,
        headers={"User-Agent": "TripPlannerBot/1.0"},
        timeout=REQUEST_TIMEOUT,
        request_name=f"OpenTripMap bbox ({lon_min},{lat_min}) offset={offset}",
        tracker=tracker,
    )

    if isinstance(data, dict):
        # Some API variants return {features:[...]} JSON.
        if "features" in data:
            features = data.get("features", [])
            normalized = []
            for f in features:
                props = f.get("properties", {})
                point = f.get("geometry", {}).get("coordinates", [None, None])
                normalized.append({
                    "xid": props.get("xid"),
                    "name": props.get("name", ""),
                    "kinds": props.get("kinds", ""),
                    "lon": point[0],
                    "lat": point[1],
                })
            return normalized

    if isinstance(data, list):
        return data

    return []


def fetch_place_details(xid, tracker=None):
    data = get_json_with_retry(
        f"{OPENTRIPMAP_API_BASE}/xid/{xid}",
        params={"apikey": OPENTRIPMAP_API_KEY},
        headers={"User-Agent": "TripPlannerBot/1.0"},
        timeout=REQUEST_TIMEOUT,
        request_name=f"OpenTripMap detail xid={xid}",
        tracker=tracker,
    )

    description = ""
    if isinstance(data.get("wikipedia_extracts"), dict):
        description = data["wikipedia_extracts"].get("text", "")

    if not description and isinstance(data.get("info"), dict):
        description = data["info"].get("descr", "") or ""

    if not description:
        address = data.get("address", {})
        country = address.get("country", "")
        city = address.get("city", "")
        state = address.get("state", "")
        description = " ".join(x for x in [city, state, country] if x)

    return {
        "xid": xid,
        "name": data.get("name", ""),
        "kinds": data.get("kinds", ""),
        "description": description,
        "lat": data.get("point", {}).get("lat"),
        "lon": data.get("point", {}).get("lon"),
        "rate": data.get("rate"),
        "raw": data,
    }


def collect_opentripmap_places(max_requests=None):
    tracker = RequestTracker(max_requests=max_requests)
    seen_xids = set()
    places = []

    tile_count = 0

    try:
        for lon_min, lon_max, lat_min, lat_max in iter_tiles(TILE_SIZE_DEGREES):
            tile_count += 1
            offset = 0

            while True:
                page = fetch_bbox_page(
                    lon_min, lon_max, lat_min, lat_max, offset=offset, tracker=tracker
                )

                if not page:
                    break

                for row in page:
                    xid = row.get("xid")
                    if not xid or xid in seen_xids:
                        continue

                    seen_xids.add(xid)

                    try:
                        details = fetch_place_details(xid, tracker=tracker)
                        # If details endpoint has no name, fallback to bbox name.
                        if not details.get("name"):
                            details["name"] = row.get("name", "")
                        if not details.get("kinds"):
                            details["kinds"] = row.get("kinds", "")
                        if details.get("lat") is None:
                            details["lat"] = row.get("point", {}).get("lat")
                        if details.get("lon") is None:
                            details["lon"] = row.get("point", {}).get("lon")

                        places.append(details)
                    except RequestBudgetExceeded:
                        raise
                    except Exception as exc:
                        print(f"Skipping xid={xid} due to detail error: {exc}")

                    time.sleep(SLEEP_BETWEEN_CALLS)

                if len(page) < LIMIT:
                    break

                offset += LIMIT
                time.sleep(SLEEP_BETWEEN_CALLS)

            if tile_count % 25 == 0:
                print(
                    f"Scanned {tile_count} tiles | "
                    f"Unique attractions collected: {len(places)} | "
                    f"HTTP calls used: {tracker.calls}"
                )

    except RequestBudgetExceeded:
        print(
            f"Stopped due to request budget: {tracker.calls}/{tracker.max_requests}"
        )

    print(
        f"Collection finished. Tiles scanned: {tile_count} | "
        f"Unique attractions: {len(places)} | HTTP calls: {tracker.calls}"
    )

    return places


# =========================
# CHUNKING
# =========================

def split_text_with_overlap(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = " ".join(text.split())

    if not text:
        return [""]

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            split_at = text.rfind(" ", start + int(chunk_size * 0.7), end)
            if split_at != -1:
                end = split_at

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(0, end - overlap)

    return chunks


def chunk_place_text(place):
    name = (place.get("name") or "Unknown").strip() or "Unknown"
    kinds = " ".join((place.get("kinds") or "").split(","))
    description = " ".join((place.get("description") or "").split())

    base = f"{name}."
    if kinds:
        base += f" Types: {kinds}."

    if not description:
        return [base]

    desc_chunks = split_text_with_overlap(description)
    return [f"{base} {chunk}" for chunk in desc_chunks]


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
    global index, texts, metadata
    index = None
    texts = []
    metadata = []


def save_outputs(output_prefix="opentripmap"):
    if output_prefix == "opentripmap":
        index_path = "opentripmap.index"
        texts_path = "opentripmap_texts.json"
        metadata_path = "opentripmap_metadata.json"
    else:
        index_path = f"{output_prefix}.index"
        texts_path = f"{output_prefix}_texts.json"
        metadata_path = f"{output_prefix}_metadata.json"

    faiss.write_index(index, index_path)

    with open(texts_path, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved index to {index_path}")
    print(f"Saved texts to {texts_path}")
    print(f"Saved metadata to {metadata_path}")


def build_and_embed(places):
    global index

    if not places:
        print("No places to embed.")
        return

    all_texts = []
    all_chunk_meta = []

    for place in places:
        chunks = chunk_place_text(place)
        chunk_count = len(chunks)

        for chunk_index, text in enumerate(chunks):
            all_texts.append(text)
            all_chunk_meta.append({
                "xid": place.get("xid"),
                "name": place.get("name", ""),
                "kinds": place.get("kinds", ""),
                "description": place.get("description", ""),
                "lat": place.get("lat"),
                "lon": place.get("lon"),
                "rate": place.get("rate"),
                "chunk_index": chunk_index,
                "chunk_count": chunk_count,
            })

    print(f"Prepared {len(all_texts)} chunks from {len(places)} attractions")

    # Batch embeddings to avoid huge payloads
    EMBED_BATCH_SIZE = 128
    start = 0

    while start < len(all_texts):
        end = min(start + EMBED_BATCH_SIZE, len(all_texts))
        batch_texts = all_texts[start:end]
        batch_meta = all_chunk_meta[start:end]

        vectors = embed_batch(batch_texts)

        for text, vec, meta_item in zip(batch_texts, vectors, batch_meta):
            if index is None:
                index = faiss.IndexFlatL2(len(vec))

            index.add(np.array([vec], dtype=np.float32))
            texts.append(text)
            metadata.append(meta_item)

        print(f"Embedded chunks: {end}/{len(all_texts)}")
        start = end


def run_pipeline(max_requests=None, output_prefix="opentripmap"):
    validate_api_key()
    reset_storage()

    started = time.time()

    places = collect_opentripmap_places(max_requests=max_requests)
    build_and_embed(places)

    elapsed = time.time() - started
    print(f"Total runtime: {elapsed:.2f}s")

    if index is None:
        print("No vectors were created; nothing to save.")
        return

    save_outputs(output_prefix=output_prefix)


def run_test(request_count=TEST_REQUESTS):
    print(f"Running OpenTripMap test mode with ~{request_count} HTTP requests...")
    run_pipeline(max_requests=request_count, output_prefix="opentripmap_test")


def run_all():
    print("Running OpenTripMap full mode (global tile scan)...")
    run_pipeline(max_requests=None, output_prefix="opentripmap")


def main():
    run_test()


if __name__ == "__main__":
    main()

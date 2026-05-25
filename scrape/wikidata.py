import time
import json
import math
import random
import requests
import numpy as np
import faiss

# =========================
# CONFIG
# =========================

WIKIDATA_URL = "https://query.wikidata.org/sparql"

LM_STUDIO_URL = "http://127.0.0.1:1234/v1/embeddings"
EMBED_MODEL = "qwen/text-embedding-qwen3-embedding-0.6b"

LIMIT = 50
MIN_LIMIT = 10
MAX_LIMIT = 100
SLEEP_BETWEEN_BATCHES = 2
REQUEST_TIMEOUT = (10, 120)
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
TEST_REQUESTS = 10
MAX_HTTP_RETRIES = 5
RETRY_BACKOFF_SECONDS = 2

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

REQUEST_HEADERS = {
    # Keep a descriptive UA for Wikidata Query Service etiquette.
    "User-Agent": "TripPlannerBot/1.0 (Wikidata ingestion pipeline)",
    "Accept": "application/sparql-results+json",
}

# =========================
# STORAGE
# =========================

index = None
texts = []
metadata = []

# =========================
# REAL WIKIDATA FETCH
# =========================

def get_json_with_retry(url, *, params, headers, timeout, request_name):
    last_error = None

    for attempt in range(1, MAX_HTTP_RETRIES + 1):
        try:
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

            if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
                retry_after = exc.response.headers.get("Retry-After")
                if retry_after:
                    try:
                        backoff = max(backoff, float(retry_after))
                    except ValueError:
                        pass

            # Add jitter to avoid synchronized retries.
            backoff += random.uniform(0, 1.0)
            print(
                f"{request_name} failed (attempt {attempt}/{MAX_HTTP_RETRIES}): {exc}. "
                f"Retrying in {backoff}s..."
            )
            time.sleep(backoff)

    # Defensive fallback (loop should always return or raise)
    raise RuntimeError(f"{request_name} failed after retries: {last_error}")

def fetch_wikidata(limit=50, last_place_uri=None):
    cursor_filter = ""
    if last_place_uri:
        cursor_filter = f"FILTER(?place > <{last_place_uri}>)"

    query = f"""
    SELECT ?place ?placeLabel ?description WHERE {{
      ?place wdt:P31/wdt:P279* wd:Q570116.
      {cursor_filter}
      OPTIONAL {{
        ?place schema:description ?description
        FILTER(LANG(?description)="en")
      }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    ORDER BY ?place
    LIMIT {limit}
    """

    data = get_json_with_retry(
        WIKIDATA_URL,
        params={"query": query, "format": "json"},
        headers=REQUEST_HEADERS,
        timeout=REQUEST_TIMEOUT,
        request_name=f"Wikidata fetch (cursor={last_place_uri}, limit={limit})"
    )

    results = []

    for item in data["results"]["bindings"]:
        results.append({
            "place": item.get("place", {}).get("value", ""),
            "name": item.get("placeLabel", {}).get("value", ""),
            "description": item.get("description", {}).get("value", "")
        })

    return results


def fetch_total_places():
    query = """
    SELECT (COUNT(DISTINCT ?place) AS ?total) WHERE {
      ?place wdt:P31/wdt:P279* wd:Q570116.
    }
    """

    data = get_json_with_retry(
        WIKIDATA_URL,
        params={"query": query, "format": "json"},
        headers=REQUEST_HEADERS,
        timeout=REQUEST_TIMEOUT,
        request_name="Wikidata total-count query"
    )

    return int(data["results"]["bindings"][0]["total"]["value"])

# =========================
# TEXT BUILDER
# =========================

def build_text(item):
    name = item.get("name", "Unknown")
    description = item.get("description", "")

    return f"{name}. {description}"


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


def chunk_poi_text(item):
    name = item.get("name", "Unknown").strip() or "Unknown"
    description = item.get("description", "")
    description = " ".join(description.split())

    if not description:
        return [f"{name}."]

    desc_chunks = split_text_with_overlap(description)
    return [f"{name}. {chunk}" for chunk in desc_chunks]

# =========================
# EMBEDDING (BATCH)
# =========================

def embed_batch(texts_batch):
    response = requests.post(
        LM_STUDIO_URL,
        json={
            "model": EMBED_MODEL,
            "input": texts_batch
        },
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    data = response.json()

    return [np.array(x["embedding"], dtype=np.float32) for x in data["data"]]

# =========================
# MAIN PIPELINE
# =========================

def reset_storage():
    global index, texts, metadata
    index = None
    texts = []
    metadata = []


def save_outputs(output_prefix="travel"):
    if output_prefix == "travel":
        index_path = "travel.index"
        texts_path = "texts.json"
        metadata_path = "metadata.json"
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


def run_pipeline(max_requests=None, output_prefix="travel"):
    global index

    reset_storage()

    start_total = time.time()
    total_places = fetch_total_places()
    max_batches = max_requests if max_requests is not None else math.ceil(total_places / LIMIT)
    target_places = min(total_places, max_batches * LIMIT)

    print(f"Total places in Wikidata match: {total_places}")
    print(f"Target places for this run: {target_places}")

    batch_idx = 0
    pois_processed = 0
    current_limit = LIMIT
    last_place_uri = None

    while True:
        if max_requests is not None and batch_idx >= max_requests:
            break

        if pois_processed >= total_places or pois_processed >= target_places:
            break

        start_fetch = time.time()

        try:
            batch_data = fetch_wikidata(limit=current_limit, last_place_uri=last_place_uri)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.HTTPError) as exc:
            current_limit = max(MIN_LIMIT, current_limit // 2)
            cooloff = SLEEP_BETWEEN_BATCHES * 3
            print(
                f"Fetch failed after retries: {exc}. "
                f"Reducing limit to {current_limit} and cooling off for {cooloff}s."
            )
            time.sleep(cooloff)
            continue

        end_fetch = time.time()

        print(
            f"\nFetched {len(batch_data)} items in {end_fetch - start_fetch:.2f}s "
            f"(cursor={last_place_uri}, limit={current_limit})"
        )

        if not batch_data:
            break

        last_place_uri = batch_data[-1].get("place") or last_place_uri

        if current_limit < MAX_LIMIT:
            current_limit = min(MAX_LIMIT, current_limit + 5)

        # -------- BUILD TEXT --------
        texts_batch = []
        chunk_metadata_batch = []

        for item in batch_data:
            poi_chunks = chunk_poi_text(item)
            chunk_count = len(poi_chunks)

            for chunk_idx, text in enumerate(poi_chunks):
                texts_batch.append(text)
                chunk_metadata_batch.append({
                    "name": item.get("name", ""),
                    "description": item.get("description", ""),
                    "chunk_index": chunk_idx,
                    "chunk_count": chunk_count
                })

        # -------- EMBED --------
        start_embed = time.time()

        embeddings = embed_batch(texts_batch)

        end_embed = time.time()

        print(
            f"Embedded {len(texts_batch)} chunks from {len(batch_data)} POIs "
            f"in {end_embed - start_embed:.2f}s"
        )

        # -------- STORE --------
        for text, vector, item in zip(texts_batch, embeddings, chunk_metadata_batch):

            if index is None:
                index = faiss.IndexFlatL2(len(vector))

            index.add(np.array([vector], dtype=np.float32))

            texts.append(text)
            metadata.append(item)

        pois_processed += len(batch_data)

        # be polite to Wikidata
        time.sleep(SLEEP_BETWEEN_BATCHES)

        batch_idx += 1

        elapsed = time.time() - start_total
        completion = (pois_processed / target_places) * 100 if target_places else 0

        if pois_processed > 0 and target_places > pois_processed:
            avg_place_seconds = elapsed / pois_processed
            remaining_places = target_places - pois_processed
            eta_seconds = remaining_places * avg_place_seconds
            eta_minutes = eta_seconds / 60
            print(
                f"Progress: {pois_processed}/{target_places} POIs ({completion:.2f}%) | "
                f"Stored chunks: {len(texts)} | "
                f"ETA: ~{eta_minutes:.1f} min"
            )

    end_total = time.time()

    print(f"\n✅ TOTAL TIME: {end_total - start_total:.2f}s")

    if index is None:
        print("No data fetched; nothing to save.")
        return

    save_outputs(output_prefix=output_prefix)


def run_test(request_count=TEST_REQUESTS):
    print(f"Running test mode with ~{request_count} requests...")
    run_pipeline(max_requests=request_count, output_prefix="travel_test")


def run_all():
    print("Running full mode (all available POIs)...")
    run_pipeline(max_requests=None, output_prefix="travel")


def main():
    run_test()


if __name__ == "__main__":
    main()

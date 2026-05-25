import json
import os
import re
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import pandas as pd
import httpx

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

CSV_PATH = os.path.join(SCRIPT_DIR, "merged-world-wikivoyage-descriptions.csv")
CITY_COL = "city"
DESC_COL = "description"

API_KEY = os.environ.get("ROUTEWAY_API_KEY", "")

CHECKPOINT_FILE = os.path.join(SCRIPT_DIR, "groq_tagged_cities_checkpoint.json")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "groq_tagged_cities.csv")
STOP_INFO_FILE = os.path.join(SCRIPT_DIR, "stop_info.json")

MODEL_NAME = "deepseek-v4-flash"
MAX_REQUESTS_PER_MIN = 30
MAX_TOKENS_PER_MIN = 10000
MAX_COMPLETION_TOKENS = 150
TEMPERATURE = 0.1
SAVE_EVERY = 25
MAX_DESCRIPTION_CHARS = 1500
MAX_RETRIES = 5
ROUTEWAY_API_URL = "https://api.routeway.ai/v1/chat/completions"

ALLOWED_TAGS = {"historic", "cultural", "religious-heritage",
"beach", "coastal", "island",
"urban", "nightlife", "food", "shopping", "luxury", "affordable",
"family-friendly", "romantic",
"nature", "adventure", "relaxation",
"skiing", "mountains",
"touristic-value", "crowded", "quiet"
}

SYSTEM_PROMPT = """Tag city for tourism. Output JSON: {"city":"<name>","tags":[{"tag":"<name>","weight":<0-1>}]}
Tags: historic, cultural, religious-heritage, beach, coastal, island, urban, nightlife, food, shopping, luxury, affordable, family-friendly, romantic, nature, adventure, relaxation, skiing, mountains, touristic-value, crowded, quiet
Rules: 7-14 tags, weight≥0.60, always include touristic-value, only tourist experience."""


class DailyLimitReached(Exception):
    pass


class SlidingWindowLimiter:
    """Simple limiter for requests/min and tokens/min."""

    def __init__(self, max_rpm: int, max_tpm: int):
        self.max_rpm = max_rpm
        self.max_tpm = max_tpm
        self.request_events = deque()
        self.token_events = deque()

    def _prune(self, now: float) -> None:
        while self.request_events and (now - self.request_events[0]) >= 60:
            self.request_events.popleft()
        while self.token_events and (now - self.token_events[0][0]) >= 60:
            self.token_events.popleft()

    def _tokens_used_last_min(self) -> int:
        return int(sum(tokens for _, tokens in self.token_events))

    def wait_for_budget(self, estimated_tokens: int) -> None:
        while True:
            now = time.time()
            self._prune(now)

            req_ok = len(self.request_events) < self.max_rpm
            tok_ok = (self._tokens_used_last_min() + estimated_tokens) <= self.max_tpm

            if req_ok and tok_ok:
                self.request_events.append(now)
                self.token_events.append((now, estimated_tokens))
                return

            req_wait = 0.0
            tok_wait = 0.0

            if not req_ok and self.request_events:
                req_wait = 60 - (now - self.request_events[0])

            if not tok_ok and self.token_events:
                tok_wait = 60 - (now - self.token_events[0][0])

            sleep_for = max(0.5, min(max(req_wait, tok_wait), 10.0))
            time.sleep(sleep_for)

    def reconcile_tokens(self, estimated: int, actual: int) -> None:
        delta = int(actual - estimated)
        if delta != 0:
            self.token_events.append((time.time(), delta))


def rough_token_estimate(text: str, max_completion_tokens: int) -> int:
    return int(len(text) / 4) + max_completion_tokens


def clean_description(description: str) -> str:
    if pd.isna(description) or str(description).strip() == "":
        return "No description available."
    text = str(description).strip()
    return text[:MAX_DESCRIPTION_CHARS]


def has_description(description: str) -> bool:
    return (not pd.isna(description)) and str(description).strip() != ""


def build_messages(city: str, description: str) -> List[Dict[str, str]]:
    user_msg = f"CITY: {city}\n\nDESCRIPTION:\n{clean_description(description)}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


def extract_json(text: str) -> Optional[Dict]:
    try:
        return json.loads(text)
    except Exception:
        pass

    candidates = re.findall(r"\{[\s\S]*\}", text)
    for candidate in reversed(candidates):
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def normalize_result(city: str, payload: Optional[Dict]) -> Dict:
    if not isinstance(payload, dict):
        return {"city": city, "tags": []}

    tags = payload.get("tags", [])
    if not isinstance(tags, list):
        return {"city": city, "tags": []}

    cleaned = []
    seen = set()
    for t in tags:
        if not isinstance(t, dict):
            continue
        name = str(t.get("tag", "")).strip()
        if name not in ALLOWED_TAGS or name in seen:
            continue
        try:
            weight = float(t.get("weight", 0.0))
        except Exception:
            continue
        weight = max(0.0, min(1.0, round(weight, 2)))
        if weight < 0.60:
            continue
        cleaned.append({"tag": name, "weight": weight})
        seen.add(name)

    if "touristic-value" not in seen:
        cleaned.append({"tag": "touristic-value", "weight": 0.70})

    return {"city": city, "tags": cleaned[:14]}


def is_daily_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    daily_limit_markers = [
        "daily",
        "per day",
        "quota",
        "exhaust",
        "limit reached",
        "billing",
    ]
    return any(marker in text for marker in daily_limit_markers)


def save_checkpoint(results: List[Dict], path: str = CHECKPOINT_FILE) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)


def load_checkpoint(path: str = CHECKPOINT_FILE) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def save_output(results: List[Dict], output_path: str = OUTPUT_FILE) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_sorted = sorted(results, key=lambda x: x.get("idx", 10**12))
    clean_results = [{"city": r.get("city", ""), "tags": r.get("tags", [])} for r in results_sorted]
    out_df = pd.DataFrame(clean_results)
    out_df.to_csv(output_path, index=False)


def save_stop_info(stop_idx: int, stop_city: str, reason: str, results_len: int) -> None:
    os.makedirs(os.path.dirname(STOP_INFO_FILE), exist_ok=True)
    payload = {
        "stopped_idx": stop_idx,
        "stopped_city": stop_city,
        "reason": reason,
        "processed_rows_including_checkpoint": results_len,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(STOP_INFO_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def tag_city_with_failover(
    limiter: SlidingWindowLimiter,
    city: str,
    description: str,
) -> Dict:
    messages = build_messages(city, description)
    prompt_text = SYSTEM_PROMPT + "\n\nCITY: " + city + "\nDESC:" + clean_description(description)[:300]
    estimated_tokens = rough_token_estimate(prompt_text, MAX_COMPLETION_TOKENS)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            limiter.wait_for_budget(estimated_tokens)

            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            }
            
            payload = {
                "model": MODEL_NAME,
                "messages": messages,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_COMPLETION_TOKENS,
                "response_format": {"type": "json_object"},
            }

            with httpx.Client(timeout=30.0) as client:
                response = client.post(ROUTEWAY_API_URL, json=payload, headers=headers)
                response.raise_for_status()
                result_data = response.json()

            content = result_data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            payload_result = extract_json(content)

            usage = result_data.get("usage", {})
            actual_tokens = usage.get("total_tokens", estimated_tokens)
            limiter.reconcile_tokens(estimated_tokens, int(actual_tokens))

            result = normalize_result(city, payload_result)
            return result

        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 429:  # Rate limit
                backoff = min(2 ** attempt, 60)
                print(f"Rate limited for {city}. Retry {attempt}/{MAX_RETRIES} in {backoff}s")
                time.sleep(backoff)
            else:
                backoff = min(2 ** attempt, 30)
                print(f"HTTP {status} error for {city}: {exc}. Retry {attempt}/{MAX_RETRIES} in {backoff}s")
                time.sleep(backoff)

        except Exception as exc:
            backoff = min(2 ** attempt, 30)
            print(f"Error for {city}: {exc}. Retry {attempt}/{MAX_RETRIES} in {backoff}s")
            time.sleep(backoff)
    
    raise DailyLimitReached(f"Failed to tag {city} after {MAX_RETRIES} retries")


def process_full_dataset(dataframe: pd.DataFrame) -> Tuple[List[Dict], Optional[Dict]]:
    results = load_checkpoint(CHECKPOINT_FILE)
    done_indices = {int(r["idx"]) for r in results if isinstance(r, dict) and "idx" in r}

    if results:
        print(f"Resuming from checkpoint with {len(results)} rows.")
    else:
        print("Starting fresh run.")

    skipped_missing_description = int(dataframe[DESC_COL].isna().sum()) if DESC_COL in dataframe.columns else 0

    remaining = [
        (idx, row[CITY_COL], row.get(DESC_COL, ""))
        for idx, row in dataframe.iterrows()
        if idx not in done_indices and has_description(row.get(DESC_COL, ""))
    ]

    print(f"Remaining rows: {len(remaining)} / {len(dataframe)}")
    if skipped_missing_description:
        print(f"Rows skipped due to missing description: {skipped_missing_description}")
    limiter = SlidingWindowLimiter(MAX_REQUESTS_PER_MIN, MAX_TOKENS_PER_MIN)

    start = time.time()
    processed_now = 0

    for idx, city, desc in remaining:
        try:
            tagged = tag_city_with_failover(limiter, city, desc)
        except DailyLimitReached as exc:
            save_checkpoint(results, CHECKPOINT_FILE)
            save_output(results, OUTPUT_FILE)
            save_stop_info(int(idx), str(city), str(exc), len(results))

            stop_payload = {
                "stopped_idx": int(idx),
                "stopped_city": str(city),
                "reason": str(exc),
            }
            print("\nStopped cleanly after both API keys reached daily limit.")
            print(f"Last unprocessed row index: {idx}")
            print(f"Last unprocessed city: {city}")
            print(f"Checkpoint saved to: {CHECKPOINT_FILE}")
            print(f"Partial output saved to: {OUTPUT_FILE}")
            print(f"Stop info saved to: {STOP_INFO_FILE}")
            return results, stop_payload

        tagged["idx"] = int(idx)
        results.append(tagged)
        processed_now += 1

        if processed_now % SAVE_EVERY == 0:
            save_checkpoint(results, CHECKPOINT_FILE)

        elapsed = time.time() - start
        rate = processed_now / max(elapsed, 1e-9)
        eta_sec = (len(remaining) - processed_now) / max(rate, 1e-9)
        print(f"[{len(dataframe)-len(remaining)+processed_now}/{len(dataframe)}] {city} | ETA: {eta_sec/60:.1f} min")

    save_checkpoint(results, CHECKPOINT_FILE)
    save_output(results, OUTPUT_FILE)
    print("Completed all rows.")
    print(f"Checkpoint saved to: {CHECKPOINT_FILE}")
    print(f"Output saved to: {OUTPUT_FILE}")
    return results, None


def main() -> None:
    if not API_KEY:
        raise ValueError("API_KEY is empty. Please add your API key to the script.")
    
    print(f"Using single API key for Routeway")
    print(f"Using model: {MODEL_NAME}")
    print(f"Max tokens per completion: {MAX_COMPLETION_TOKENS}")

    df = pd.read_csv(CSV_PATH)
    print(f"Rows: {len(df)}")
    print("Columns:", df.columns.tolist())

    _, stop_payload = process_full_dataset(df)

    if stop_payload is None:
        print("Run finished successfully.")


if __name__ == "__main__":
    main()

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

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

CHECKPOINT_FILE = os.path.join(SCRIPT_DIR, "deepseek_tagged_cities_checkpoint.json")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "deepseek_tagged_cities.csv")
STOP_INFO_FILE = os.path.join(SCRIPT_DIR, "deepseek_stop_info.json")
ERROR_FILE = os.path.join(SCRIPT_DIR, "deepseek_errors.csv")

MODEL_NAME = "deepseek-chat"
MAX_REQUESTS_PER_MIN = 60
MAX_TOKENS_PER_MIN = 10000
MAX_COMPLETION_TOKENS = 500  # Increased from 150 to avoid truncation on 'finish_reason'='length'
TEMPERATURE = 0.1
SAVE_EVERY = 25
MAX_DESCRIPTION_CHARS = 1500
MAX_RETRIES = 5
MAX_INVALID_RETRIES = 5
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

ALLOWED_TAGS = {"historic", "cultural", "religious-heritage",
"beach", "coastal", "island",
"urban", "nightlife", "food", "shopping", "luxury", "affordable",
"family-friendly", "romantic",
"nature", "adventure", "relaxation",
"skiing", "mountains",
"touristic-value", "crowded", "quiet"
}

SYSTEM_PROMPT = """You are a tourism perception tagging system.

Given a city name and optional description, output a JSON object describing how tourists actually perceive and experience this destination.

Focus ONLY on tourist experience, expectations, and activities.
Ignore administrative facts (e.g., whether it is a capital).

OUTPUT FORMAT (valid JSON only, no markdown):

{
"city": "<city_name>",
"tags": [
{ "tag": "<tag_name>", "weight": <float_between_0_and_1> }
]
}

ALLOWED TAGS (only these):

["historic", "cultural", "religious-heritage",
"beach", "coastal", "island",
"urban", "nightlife", "food", "shopping", "luxury", "affordable",
"family-friendly", "romantic",
"nature", "adventure", "relaxation",
"skiing", "mountains",
"touristic-value", "crowded", "quiet"]

CORE PRINCIPLES:
- Think like a first-time international tourist
- Reflect what people travel there for, not everything that exists
- Prioritize experience over availability
- Example: nearby mountains ≠ \"nature\" unless tourists actively feel it
- Avoid overrating lesser-known or niche destinations

TAGGING RULES:
- Always include \"touristic-value\"
- Return 7 to 14 tags total
- Only include tags with weight ≥ 0.60
- Weight meaning:
  - 0.90–1.00 → defining feature (primary reason to visit)
  - 0.75–0.89 → strong draw
  - 0.60–0.74 → secondary but noticeable
- Do NOT include weak or irrelevant tags

CONSISTENCY RULES:
- \"touristic-value\" calibration:
  - 0.90–1.00 → globally iconic destinations
  - 0.75–0.89 → well-known tourist cities
  - 0.60–0.74 → regional or niche destinations
- Never exceed realistic global appeal

GEOGRAPHY CONSTRAINTS:
- No \"beach\" or \"coastal\" for landlocked cities
- Use \"mountains\" and/or \"skiing\" when relevant
- Use \"island\" only if it strongly shapes the experience

CORRELATION RULES:
- \"luxury\" ↑ → usually \"affordable\" ↓
- \"crowded\" ↔ high \"touristic-value\"
- \"quiet\" ↔ lower \"touristic-value\"
- \"family-friendly\" applies when clearly a major appeal (theme parks, safety, resorts)

IMPORTANT EDGE CASES:
- Do NOT overestimate small, quiet, or less-visited places
- Do NOT inflate \"cultural\" or \"historic\" unless they are a primary tourist draw
- Include \"adventure\" when activities (desert safari, skiing, hiking, etc.) are a real attraction
- Include \"relaxation\" when spas, resorts, or calm atmosphere are noticeable

GOAL:
Produce tags that are accurate, consistent across cities, and aligned with real tourist expectations and choices."""


class DailyLimitReached(Exception):
    pass


class InvalidResponse(Exception):
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
    """Normalize result. Raises InvalidResponse if response is invalid.
    """
    
    if not isinstance(payload, dict):
        raise InvalidResponse(f"Response is not a dict for {city}")

    tags = payload.get("tags", [])
    if not isinstance(tags, list):
        raise InvalidResponse(f"Tags field is not a list for {city}")

    # If tags list is empty, that's invalid
    if not tags:
        raise InvalidResponse(f"Empty tags list for {city}")

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

    # If no valid tags were extracted, that's invalid
    if not cleaned:
        raise InvalidResponse(f"No valid tags extracted for {city}")

    # Ensure touristic-value is always present
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


def save_error(city: str, country: str, error_type: str, detailed_error: str, attempt: int = 1, exact_response: str = "") -> None:
    """Append error entry to errors CSV with detailed error information and exact API response."""
    os.makedirs(os.path.dirname(ERROR_FILE), exist_ok=True)
    error_data = [{
        "city": city,
        "country": country,
        "error_type": error_type,
        "error_detail": detailed_error,
        "exact_response": exact_response[:1000],  # Truncate to 1000 chars for CSV readability
        "attempt": attempt,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }]
    error_df = pd.DataFrame(error_data)
    
    if os.path.exists(ERROR_FILE):
        existing_df = pd.read_csv(ERROR_FILE)
        error_df = pd.concat([existing_df, error_df], ignore_index=True)
    
    error_df.to_csv(ERROR_FILE, index=False)


def tag_city_with_failover(
    limiter: SlidingWindowLimiter,
    city: str,
    description: str,
) -> Dict:
    messages = build_messages(city, description)
    prompt_text = SYSTEM_PROMPT + "\n\nCITY: " + city + "\nDESC:" + clean_description(description)[:300]
    estimated_tokens = rough_token_estimate(prompt_text, MAX_COMPLETION_TOKENS)
    last_response_text = ""  # Store raw API response for error logging

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
                response = client.post(DEEPSEEK_API_URL, json=payload, headers=headers)
                response.raise_for_status()
                result_data = response.json()

            # Capture raw response for debugging
            last_response_text = str(result_data)
            
            # Check if response was truncated due to token limit
            finish_reason = result_data.get("choices", [{}])[0].get("finish_reason", "stop")
            if finish_reason == "length":
                raise InvalidResponse(
                    f"Response truncated (finish_reason='length') - increase MAX_COMPLETION_TOKENS. "
                    f"Model stopped early for {city}"
                )
            
            content = result_data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            payload_result = extract_json(content)

            usage = result_data.get("usage", {})
            actual_tokens = usage.get("total_tokens", estimated_tokens)
            limiter.reconcile_tokens(estimated_tokens, int(actual_tokens))

            result = normalize_result(city, payload_result)
            return result

        except InvalidResponse as exc:
            if attempt < MAX_INVALID_RETRIES:
                backoff = min(2 ** attempt, 15)
                error_msg = f"Invalid response for {city}: {exc}. Retry {attempt}/{MAX_INVALID_RETRIES} in {backoff}s"
                print(error_msg)
                time.sleep(backoff)
            else:
                # Attach the last response to the exception for logging
                exc.last_response = last_response_text
                raise

        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            exact_response = str(exc.response.text) if hasattr(exc.response, 'text') else str(exc)
            error_body = exact_response[:500]
            if status == 429:  # Rate limit
                backoff = min(2 ** attempt, 60)
                print(f"Rate limited for {city}. Retry {attempt}/{MAX_RETRIES} in {backoff}s")
                time.sleep(backoff)
            else:
                backoff = min(2 ** attempt, 30)
                error_msg = f"HTTP {status} error for {city}: {error_body}. Retry {attempt}/{MAX_RETRIES} in {backoff}s"
                print(error_msg)
                time.sleep(backoff)

        except Exception as exc:
            backoff = min(2 ** attempt, 30)
            error_type = type(exc).__name__
            error_detail = f"[{error_type}] {str(exc)[:300]}"
            error_msg = f"Error for {city}: {error_detail}. Retry {attempt}/{MAX_RETRIES} in {backoff}s"
            print(error_msg)
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

    # Get country column if it exists, default to empty string
    country_col = "country" if "country" in dataframe.columns else None
    
    remaining = [
        (idx, row[CITY_COL], row.get(DESC_COL, ""), row.get(country_col, "") if country_col else "")
        for idx, row in dataframe.iterrows()
        if idx not in done_indices and has_description(row.get(DESC_COL, ""))
    ]

    print(f"Remaining rows: {len(remaining)} / {len(dataframe)}")
    if skipped_missing_description:
        print(f"Rows skipped due to missing description: {skipped_missing_description}")
    limiter = SlidingWindowLimiter(MAX_REQUESTS_PER_MIN, MAX_TOKENS_PER_MIN)

    start = time.time()
    processed_now = 0
    skipped_now = 0

    for idx, city, desc, country in remaining:
        try:
            tagged = tag_city_with_failover(limiter, city, desc)
        except InvalidResponse as exc:
            skipped_now += 1
            error_msg = str(exc)
            # Extract the exact response that was attached to the exception
            exact_response = getattr(exc, 'last_response', error_msg)
            print(f"Skipping {city} ({country}): Invalid response after retries - {error_msg}")
            save_error(city, country, "InvalidResponse", error_msg, attempt=MAX_INVALID_RETRIES, exact_response=str(exact_response))
            continue
        except DailyLimitReached as exc:
            save_checkpoint(results, CHECKPOINT_FILE)
            save_output(results, OUTPUT_FILE)
            save_stop_info(int(idx), str(city), str(exc), len(results))

            stop_payload = {
                "stopped_idx": int(idx),
                "stopped_city": str(city),
                "reason": str(exc),
            }
            print("\nStopped cleanly after reaching daily limit.")
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
        eta_sec = (len(remaining) - processed_now - skipped_now) / max(rate, 1e-9)
        print(f"[{len(dataframe)-len(remaining)+processed_now+skipped_now}/{len(dataframe)}] {city} ({country}) | ETA: {eta_sec/60:.1f} min")

    save_checkpoint(results, CHECKPOINT_FILE)
    save_output(results, OUTPUT_FILE)
    print("Completed all rows.")
    print(f"Processed: {processed_now}, Skipped (invalid): {skipped_now}")
    print(f"Checkpoint saved to: {CHECKPOINT_FILE}")
    print(f"Output saved to: {OUTPUT_FILE}")
    if skipped_now > 0:
        print(f"Errors saved to: {ERROR_FILE}")
    return results, None


def main() -> None:
    if not API_KEY:
        raise ValueError("API_KEY is empty. Please set DEEPSEEK_API_KEY environment variable.")
    
    print(f"Using Deepseek API")
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

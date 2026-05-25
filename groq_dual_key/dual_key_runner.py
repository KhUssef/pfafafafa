import json
import os
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
from groq import APIError, Groq, RateLimitError


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

CSV_PATH = os.path.join(PROJECT_ROOT, "world-wikivoyage-descriptions-redo.csv")
CITY_COL = "subcountry"
DESC_COL = "description"

API_KEYS = [
    "key1_here",
    "key2_here",
    "key3_here",
    "key4_here",
    "key5_here",
    "key6_here",
    "key7_here",
    "key8_here",
    "key9_here",
    "key10_here",
]

CHECKPOINT_FILE = os.path.join(SCRIPT_DIR, "groq_tagged_cities_checkpoint.json")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "groq_tagged_cities.csv")
STOP_INFO_FILE = os.path.join(SCRIPT_DIR, "stop_info.json")

MODEL_NAME = "llama-3.3-70b-versatile"
MAX_REQUESTS_PER_MIN = 20
MAX_TOKENS_PER_MIN = 6000
MAX_COMPLETION_TOKENS = 220
TEMPERATURE = 0.1
SAVE_EVERY = 25
MAX_DESCRIPTION_CHARS = 2200
MAX_RETRIES = 6

ALLOWED_TAGS = {"historic", "cultural", "religious-heritage",
"beach", "coastal", "island",
"urban", "nightlife", "food", "shopping", "luxury", "affordable",
"family-friendly", "romantic",
"nature", "adventure", "relaxation",
"skiing", "mountains",
"touristic-value", "crowded", "quiet"
}

SYSTEM_PROMPT = """You are a tourism perception tagging system.

Given a city name (and optional description), output a JSON object describing how tourists actually perceive and experience this destination.

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
Think like a first-time international tourist
Reflect what people travel there for, not everything that exists
Prioritize experience over availability
Example: nearby mountains ≠ "nature" unless tourists actively feel it
Avoid overrating lesser-known or niche destinations
TAGGING RULES:
Always include "touristic-value"
Return 7 to 14 tags total
Only include tags with weight ≥ 0.60
Weight meaning:
0.90–1.00 → defining feature (primary reason to visit)
0.75–0.89 → strong draw
0.60–0.74 → secondary but noticeable
Do NOT include weak or irrelevant tags
CONSISTENCY RULES:
"touristic-value" calibration:
0.90–1.00 → globally iconic destinations
0.75–0.89 → well-known tourist cities
0.60–0.74 → regional or niche destinations
Never exceed realistic global appeal
Geography constraints:
No "beach" or "coastal" for landlocked cities
Use "mountains" and/or "skiing" when relevant
Use "island" only if it strongly shapes the experience
Correlation rules:
"luxury" ↑ → usually "affordable" ↓
"crowded" ↔ high "touristic-value"
"quiet" ↔ lower "touristic-value"
"family-friendly" applies when clearly a major appeal (theme parks, safety, resorts)
IMPORTANT EDGE CASES:
Do NOT overestimate:
Small, quiet, or less-visited places
Cities with limited global recognition
Do NOT inflate "cultural" or "historic" unless they are a primary tourist draw
Include "adventure" when activities (desert safari, skiing, hiking, etc.) are a real attraction
Include "relaxation" when spas, resorts, or calm atmosphere are noticeable
GOAL:

Produce tags that are:

Accurate
Consistent across cities
Aligned with real tourist expectations and choices
"""


class DailyLimitReached(Exception):
    pass


class AllKeysExhausted(Exception):
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


def get_retry_after_seconds(exc: Exception) -> float:
    response = getattr(exc, "response", None)
    if response is None:
        return 0.0
    header = response.headers.get("retry-after") if getattr(response, "headers", None) else None
    if not header:
        return 0.0
    try:
        return float(header)
    except Exception:
        return 0.0


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


@dataclass
class GroqKeyState:
    name: str
    key: str
    client: Groq


class MultiKeyGroqClient:
    def __init__(self, keys: List[str]):
        if not keys:
            raise ValueError("At least one API key is required.")
        
        self.keys: List[GroqKeyState] = [
            GroqKeyState(
                name=f"KEY_{i}",
                key=key,
                client=Groq(api_key=key),
            )
            for i, key in enumerate(keys)
        ]
        self.active_index = 0
        print(f"Initialized {len(self.keys)} API keys for rotation.")

    @property
    def active(self) -> GroqKeyState:
        if not self.keys:
            raise AllKeysExhausted("No API keys remaining.")
        return self.keys[self.active_index]
    
    def has_keys(self) -> bool:
        """Check if there are any keys left."""
        return len(self.keys) > 0

    def rotate_to_next_key(self) -> None:
        """Rotate to next key in circular fashion."""
        if self.keys:
            self.active_index = (self.active_index + 1) % len(self.keys)
            print(f"Rotated to key: {self.active.name}")

    def remove_current_key(self) -> None:
        """Remove the current exhausted key from the list."""
        if not self.keys:
            return
        
        removed_key = self.keys[self.active_index].name
        self.keys.pop(self.active_index)
        print(f"Removed {removed_key}. {len(self.keys)} key(s) remaining.")
        
        if not self.keys:
            raise AllKeysExhausted("All API keys have been exhausted.")
        
        if self.active_index >= len(self.keys):
            self.active_index = 0

    def mark_active_exhausted_and_rotate(self) -> None:
        """Remove current key and rotate to next available key."""
        self.remove_current_key()


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
    multi_client: MultiKeyGroqClient,
    limiter: SlidingWindowLimiter,
    city: str,
    description: str,
) -> Dict:
    messages = build_messages(city, description)
    prompt_text = SYSTEM_PROMPT + "\n\nCITY: " + city + "\nDESCRIPTION:\n" + clean_description(description)
    estimated_tokens = rough_token_estimate(prompt_text, MAX_COMPLETION_TOKENS)

    while True:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                limiter.wait_for_budget(estimated_tokens)

                response = multi_client.active.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_completion_tokens=MAX_COMPLETION_TOKENS,
                    response_format={"type": "json_object"},
                )

                content = response.choices[0].message.content or "{}"
                payload = extract_json(content)

                actual_tokens = getattr(getattr(response, "usage", None), "total_tokens", None)
                if actual_tokens is not None:
                    limiter.reconcile_tokens(estimated_tokens, int(actual_tokens))

                result = normalize_result(city, payload)
                
                # Rotate to next key after successful request
                if multi_client.has_keys():
                    multi_client.rotate_to_next_key()
                
                return result

            except RateLimitError as exc:
                if is_daily_limit_error(exc):
                    print(f"Daily limit hit on {multi_client.active.name} while processing {city}.")
                    try:
                        multi_client.mark_active_exhausted_and_rotate()
                        break
                    except AllKeysExhausted as all_exc:
                        raise DailyLimitReached(str(all_exc)) from all_exc

                retry_after = get_retry_after_seconds(exc)
                backoff = min(2 ** attempt, 60)
                sleep_for = max(retry_after, backoff)
                print(
                    f"Rate limited for {city} on {multi_client.active.name}. "
                    f"Retry {attempt}/{MAX_RETRIES} in {sleep_for:.1f}s"
                )
                time.sleep(sleep_for)

            except APIError as exc:
                backoff = min(2 ** attempt, 45)
                print(f"API error for {city}: {exc}. Retry {attempt}/{MAX_RETRIES} in {backoff}s")
                time.sleep(backoff)

            except Exception as exc:
                backoff = min(2 ** attempt, 30)
                print(f"Unexpected error for {city}: {exc}. Retry {attempt}/{MAX_RETRIES} in {backoff}s")
                time.sleep(backoff)
        else:
            return {"city": city, "tags": []}


def process_full_dataset(dataframe: pd.DataFrame, multi_client: MultiKeyGroqClient) -> Tuple[List[Dict], Optional[Dict]]:
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
            tagged = tag_city_with_failover(multi_client, limiter, city, desc)
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
    if not API_KEYS:
        raise ValueError("API_KEYS list is empty. Please add your API keys to the script.")
    
    print(f"Loaded {len(API_KEYS)} API key(s) for rotation.")

    df = pd.read_csv(CSV_PATH)
    print(f"Rows: {len(df)}")
    print("Columns:", df.columns.tolist())

    multi_client = MultiKeyGroqClient(API_KEYS)
    _, stop_payload = process_full_dataset(df, multi_client)

    if stop_payload is None:
        print("Run finished without exhausting all API keys.")


if __name__ == "__main__":
    main()

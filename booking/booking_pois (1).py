"""Scrape Booking.com Attractions for POIs near a city search.

Extracts rich structured data from each attraction card:
  name, url, location, description, rating, review_count, rating_label,
  price_from, currency, availability, image_alt, image_url.

Supports optional constraints:
  --max-price     Filter by maximum price per person (in page currency)
  --min-rating    Filter by minimum rating (e.g. 4.0)
  --adults        Number of adults (passed to search URL)
  --group-size    Synonym for adults

Usage examples:
    python booking_pois.py --country France --city Paris
    python booking_pois.py --country France --city Paris --checkin 2026-05-01 --checkout 2026-05-02
    python booking_pois.py --country UAE --city Dubai --max-price 50 --min-rating 4.3
    python booking_pois.py --country "United Arab Emirates" --city "Abu Dhabi" --output pois.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from html import unescape
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import ProxyHandler, Request, build_opener, urlopen


AUTOCOMPLETE_URL = "https://accommodations.booking.com/autocomplete.json"
SEARCH_URL = "https://www.booking.com/searchresults.en-gb.html"
ATTRACTIONS_LANDING_URL = "https://www.booking.com/attractions/index.en-gb.html"
BOOKING_BASE = "https://www.booking.com"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
DEFAULT_PROXY_FILE = Path(__file__).resolve().parents[1] / "scrape" / "Webshare 10 proxies.txt"
DEFAULT_SESSION_COOKIE_FILE = Path(__file__).resolve().parent / "booking_cookies.json"


# ---------------------------------------------------------------------------
# Proxy / cookie helpers (unchanged from original)
# ---------------------------------------------------------------------------

def _parse_proxy_line(line: str) -> Optional[str]:
    raw = line.strip()
    if not raw or raw.startswith("#"):
        return None
    parts = raw.split(":")
    if len(parts) == 2:
        host, port = parts
        return f"http://{host}:{port}"
    if len(parts) >= 4:
        host = parts[0]
        port = parts[1]
        user = quote(parts[2], safe="")
        password = quote(":".join(parts[3:]), safe="")
        return f"http://{user}:{password}@{host}:{port}"
    return None


def _proxy_label(proxy_url: Optional[str]) -> str:
    if not proxy_url:
        return "direct"
    split = urlsplit(proxy_url)
    if split.hostname and split.port:
        return f"http://{split.hostname}:{split.port}"
    return "proxy"


def load_proxies(proxy_file: Optional[str]) -> List[str]:
    if not proxy_file:
        return []
    proxy_path = Path(proxy_file)
    if not proxy_path.exists():
        return []
    proxies: List[str] = []
    for line in proxy_path.read_text(encoding="utf-8", errors="replace").splitlines():
        proxy = _parse_proxy_line(line)
        if proxy:
            proxies.append(proxy)
    return proxies


def _load_cookie_records(cookie_file: Optional[str]) -> List[dict]:
    if not cookie_file:
        return []
    cookie_path = Path(cookie_file)
    if not cookie_path.exists():
        return []
    raw_text = cookie_path.read_text(encoding="utf-8", errors="replace").strip()
    if not raw_text:
        return []
    if raw_text.startswith("["):
        data = json.loads(raw_text)
        return [item for item in data if isinstance(item, dict) and item.get("name") and item.get("value")]
    cookies: List[dict] = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "\t" in stripped:
            parts = stripped.split("\t")
            if len(parts) >= 7:
                name = parts[5]
                value = parts[6]
                domain = parts[0].lstrip("#")
                path = parts[2] or "/"
                cookies.append({"name": name, "value": value, "domain": domain, "path": path})
                continue
        if "=" in stripped:
            name, value = stripped.split("=", 1)
            cookies.append({"name": name.strip(), "value": value.strip(), "path": "/"})
    return cookies


def _cookie_header(cookie_records: List[dict]) -> str:
    morsels = []
    for cookie in cookie_records:
        name = str(cookie.get("name", "")).strip()
        value = str(cookie.get("value", ""))
        if name:
            morsels.append(f"{name}={value}")
    return "; ".join(morsels)


def _load_cookie_header(cookie_file: Optional[str]) -> str:
    return _cookie_header(_load_cookie_records(cookie_file))


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _open_request(request: Request, proxy_url: Optional[str], timeout: int):
    if proxy_url:
        handler = ProxyHandler({"http": proxy_url, "https": proxy_url})
        opener = build_opener(handler)
        return opener.open(request, timeout=timeout)
    return urlopen(request, timeout=timeout)


def _http_json(
    url: str,
    payload: Optional[dict] = None,
    headers: Optional[dict] = None,
    proxy_url: Optional[str] = None,
    cookie_header: str = "",
) -> dict:
    request_headers = {
        "Accept": "application/json, text/plain, */*",
        "Content-Type": "application/json",
        "Origin": "https://www.booking.com",
        "Referer": "https://www.booking.com/",
        "User-Agent": USER_AGENT,
    }
    if headers:
        request_headers.update(headers)
    if cookie_header:
        request_headers["Cookie"] = cookie_header

    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    request = Request(url, data=data, headers=request_headers, method="POST" if data else "GET")
    with _open_request(request, proxy_url=proxy_url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def _http_text(
    url: str,
    headers: Optional[dict] = None,
    proxy_url: Optional[str] = None,
    cookie_header: str = "",
) -> str:
    request_headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Upgrade-Insecure-Requests": "1",
        "User-Agent": USER_AGENT,
    }
    if headers:
        request_headers.update(headers)
    if cookie_header:
        request_headers["Cookie"] = cookie_header

    request = Request(url, headers=request_headers)
    with _open_request(request, proxy_url=proxy_url, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# HTML utilities
# ---------------------------------------------------------------------------

def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _strip_tags(html_text: str) -> str:
    text = re.sub(r"<script\b.*?</script>", " ", html_text, flags=re.S | re.I)
    text = re.sub(r"<style\b.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<svg\b.*?</svg>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return _normalize_whitespace(unescape(text))


def _attr(html_fragment: str, attr_name: str) -> Optional[str]:
    """Extract first occurrence of attr_name="value" or attr_name='value'."""
    m = re.search(rf'{re.escape(attr_name)}=["\']([^"\']*)["\']', html_fragment, re.I)
    return unescape(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Block detection
# ---------------------------------------------------------------------------

def _looks_blocked(html_text: str) -> bool:
    lowered = html_text.lower()
    hard_blockers = [
        "verify that you're not a robot",
        "enable javascript and disable any ad blocker",
    ]
    if any(marker in lowered for marker in hard_blockers):
        return True
    soft_markers = [
        "window.awswafcookiedomainlist",
        "chal_report",
        "reportchallengeerror",
    ]
    score = sum(1 for marker in soft_markers if marker in lowered)
    has_empty_title = "<title></title>" in lowered
    return score >= 2 and has_empty_title


def _looks_empty_or_invalid_html(html_text: str) -> bool:
    lowered = html_text.lower()
    if len(html_text.strip()) < 200:
        return True
    return "<html" not in lowered or "<body" not in lowered


# ---------------------------------------------------------------------------
# Rich card extraction  ← main improvement
# ---------------------------------------------------------------------------

def _parse_price(raw: str) -> Tuple[Optional[float], Optional[str]]:
    """Return (amount, currency_symbol) or (None, None) if unparseable."""
    # Remove &nbsp; and similar
    cleaned = unescape(raw).replace("\xa0", " ").strip()
    # Match leading currency symbol then digits, or digits then symbol
    m = re.match(r"^([^\d\s]{1,3})\s*([\d,\.]+)$", cleaned)
    if m:
        symbol, amount_str = m.group(1), m.group(2)
        amount_str = amount_str.replace(",", "")
        try:
            return float(amount_str), symbol
        except ValueError:
            pass
    m = re.match(r"^([\d,\.]+)\s*([^\d\s]{1,3})$", cleaned)
    if m:
        amount_str, symbol = m.group(1), m.group(2)
        amount_str = amount_str.replace(",", "")
        try:
            return float(amount_str), symbol
        except ValueError:
            pass
    return None, None


def _extract_attr_results_sections(html_text: str) -> List[str]:
    """Return the innerHTML of every attr-search-results-page-main-content div."""
    sections: List[str] = []
    start_pattern = re.compile(
        r'<div\b[^>]*(?:id|class)="[^"]*\battr-search-results-page-main-content\b[^"]*"[^>]*>',
        flags=re.I,
    )
    div_tag_pattern = re.compile(r"</?div\b[^>]*>", flags=re.I)

    for start_match in start_pattern.finditer(html_text):
        content_start = start_match.end()
        depth = 1
        content_end = None
        for div_tag in div_tag_pattern.finditer(html_text, pos=content_start):
            tag_text = div_tag.group(0)
            if tag_text.startswith("</"):
                depth -= 1
                if depth == 0:
                    content_end = div_tag.start()
                    break
            else:
                depth += 1
        if content_end is not None and content_end > content_start:
            sections.append(html_text[content_start:content_end])

    return sections


def _extract_poi_li_html(html_text: str) -> List[str]:
    """Return raw HTML of each product-card <li> block (deduplicated)."""
    poi_blocks: List[str] = []
    seen: set = set()

    # First try the scoped results section; fall back to full page.
    sections = _extract_attr_results_sections(html_text) or [html_text]

    for section_html in sections:
        for match in re.finditer(r"<li\b[^>]*>[\s\S]*?</li>", section_html, flags=re.I):
            block = match.group(0).strip()
            if not block:
                continue
            if re.search(r"\bdata-attr-sr-banner\s*=\s*['\"]?true", block, flags=re.I):
                continue
            if not re.search(r"\bdata-product-card\s*=\s*['\"]?true", block, flags=re.I):
                continue
            key = re.sub(r"\s+", " ", block).strip().lower()
            if key in seen:
                continue
            seen.add(key)
            poi_blocks.append(block)

    return poi_blocks


def _parse_card(block: str) -> Optional[dict]:
    """
    Parse a single product-card <li> block into a rich dict.

    Returns None if the card has no recognisable title.
    """
    # ---- title & URL -------------------------------------------------------
    title_match = re.search(
        r'<h3\b[^>]*data-testid=["\']card-title["\'][^>]*>([\s\S]*?)</h3>',
        block, flags=re.I,
    )
    if not title_match:
        title_match = re.search(
            r'<a\b[^>]*href=["\'][^"\']*\/attractions\/[^"\']*["\'][^>]*>([\s\S]*?)</a>',
            block, flags=re.I,
        )
    if not title_match:
        return None

    name = _strip_tags(title_match.group(1))
    if not name:
        return None

    # URL: prefer the href in the card-title anchor
    href_match = re.search(
        r'data-testid=["\']card-title["\'][^>]*>[\s\S]*?href=["\']([^"\']+)["\']',
        block, flags=re.I,
    )
    if not href_match:
        href_match = re.search(r'href=["\'](/attractions/[^"\']+)["\']', block, flags=re.I)
    url = None
    if href_match:
        raw_href = unescape(href_match.group(1))
        url = BOOKING_BASE + raw_href if raw_href.startswith("/") else raw_href

    # ---- location ----------------------------------------------------------
    # The city label sits in the first css-1utx3w7 div
    location_match = re.search(r'css-1utx3w7[^>]*>([^<]+)', block, re.I)
    location = unescape(location_match.group(1).strip()) if location_match else None

    # ---- description -------------------------------------------------------
    desc_match = re.search(r'css-1usy0qg[^>]*>([\s\S]*?)</div>', block, re.I)
    description = _strip_tags(desc_match.group(1)) if desc_match else None

    # ---- rating & review count ---------------------------------------------
    rating: Optional[float] = None
    review_count: Optional[int] = None
    rating_label: Optional[str] = None

    # Accessible text: "User reviews, 4.4 out of 5 stars from 3216 reviews"
    review_text_match = re.search(
        r'User reviews,\s*([\d.]+)\s*out of 5 stars from\s*([\d,]+)\s*reviews',
        block, flags=re.I,
    )
    if review_text_match:
        try:
            rating = float(review_text_match.group(1))
        except ValueError:
            pass
        try:
            review_count = int(review_text_match.group(2).replace(",", ""))
        except ValueError:
            pass

    # Rating label: "Fabulous", "Superb", etc. — between "·" and first "("
    label_match = re.search(r'·\s*</span>\s*([A-Za-z][A-Za-z\s]+?)\s*</span>', block)
    if label_match:
        rating_label = label_match.group(1).strip()

    # ---- price -------------------------------------------------------------
    price_from: Optional[float] = None
    currency: Optional[str] = None

    # Accessible text: "Current price from €\xa05"
    price_text_match = re.search(r'Current price from\s*([^\s<][^<]*)', block, re.I)
    if price_text_match:
        price_from, currency = _parse_price(price_text_match.group(1))

    # ---- availability ------------------------------------------------------
    avail_match = re.search(r'fff1944c52 fb14de7f14[^>]*>([^<]+)', block, re.I)
    availability = avail_match.group(1).strip() if avail_match else None

    # ---- image -------------------------------------------------------------
    img_match = re.search(r'<img\b[^>]*alt="([^"]*)"[^>]*src="([^"]+)"', block, re.I)
    image_alt = img_match.group(1) if img_match else None
    image_url = img_match.group(2) if img_match else None

    return {
        "name": name,
        "url": url,
        "location": location,
        "description": description,
        "rating": rating,
        "rating_label": rating_label,
        "review_count": review_count,
        "price_from": price_from,
        "currency": currency,
        "availability": availability,
        "image_alt": image_alt,
        "image_url": image_url,
    }


# ---------------------------------------------------------------------------
# Constraint filtering
# ---------------------------------------------------------------------------

def _apply_constraints(
    cards: List[dict],
    min_rating: Optional[float] = None,
    max_price: Optional[float] = None,
) -> List[dict]:
    """Return only cards that satisfy rating and/or price constraints."""
    result = []
    for card in cards:
        if min_rating is not None:
            if card["rating"] is None or card["rating"] < min_rating:
                continue
        if max_price is not None:
            if card["price_from"] is None or card["price_from"] > max_price:
                continue
        result.append(card)
    return result


# ---------------------------------------------------------------------------
# Legacy POI helpers (kept for --legacy-pois flag / backward compat)
# ---------------------------------------------------------------------------

def _normalize_whitespace_legacy(text: str) -> str:
    return _normalize_whitespace(text)


def _extract_landmarks_from_html(html_text: str) -> List[str]:
    landmarks: List[str] = []
    seen: set = set()

    for match in re.finditer(
        r'data-filters-item="popular_nearby_landmarks:[^"]*"[\s\S]{0,240}?aria-label="([^"]+?):\s*\d+\s+properties"',
        html_text, flags=re.I,
    ):
        candidate = _normalize_whitespace(unescape(match.group(1)))
        if candidate and candidate.lower() not in seen:
            seen.add(candidate.lower())
            landmarks.append(candidate)

    for match in re.finditer(
        r'"urlId":"popular_nearby_landmarks=\d+"[\s\S]{0,260}?"text":"([^"]+)"',
        html_text, flags=re.I,
    ):
        candidate = _normalize_whitespace(unescape(match.group(1).replace("\\u0026", "&")))
        if candidate and candidate.lower() not in seen:
            seen.add(candidate.lower())
            landmarks.append(candidate)

    return landmarks


def _clean_candidate(candidate: str, city: str, country: str) -> Optional[str]:
    cleaned = _normalize_whitespace(unescape(unescape(candidate)))
    cleaned = cleaned.strip(" ,.;:-")
    cleaned = re.sub(r"\b(?:hotel|hostel|apartment|apartments|resort|spa|collection|city|district)\b.*$", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\b\d+\s*br\b", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\b(?:bedroom|suite|luxury|premium|ultra-luxury|private room)\b", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\b(?:metro|station)\b", "", cleaned, flags=re.I)
    cleaned = _normalize_whitespace(cleaned)
    if not cleaned:
        return None
    blocked_exact = {city.lower(), country.lower(), f"{city}, {country}".lower()}
    if cleaned.lower() in blocked_exact:
        return None
    blocked_fragments = (
        "make an informed decision", "highly rated", "featured", "value for money",
        "show on map", "opens in new window", "properties found", "reviews",
        "free cancellation", "breakfast", "private bathroom", "air conditioning",
        "swimming pool", "hot tub", "kitchen", "parking", "stars", "hotel",
        "guest house", "holiday home",
    )
    lowered = cleaned.lower()
    if any(fragment in lowered for fragment in blocked_fragments):
        return None
    if len(cleaned) < 3:
        return None
    return cleaned


# ---------------------------------------------------------------------------
# Main extraction entry point
# ---------------------------------------------------------------------------

def extract_attractions(
    html_text: str,
    city: str,
    country: str,
    limit: int = 50,
    min_rating: Optional[float] = None,
    max_price: Optional[float] = None,
) -> List[dict]:
    """
    Parse all attraction product cards from the page and return rich dicts.

    Applies optional constraints (min_rating, max_price) before returning.
    Results are sorted by rating descending (unrated last).
    """
    raw_cards: List[dict] = []
    seen_names: set = set()

    for block in _extract_poi_li_html(html_text):
        card = _parse_card(block)
        if card is None:
            continue
        key = card["name"].casefold()
        if key in seen_names:
            continue
        seen_names.add(key)
        raw_cards.append(card)

    filtered = _apply_constraints(raw_cards, min_rating=min_rating, max_price=max_price)

    # Sort: rated cards first (highest rating first), then unrated
    def sort_key(c: dict):
        r = c.get("rating")
        return (0 if r is None else 1, r or 0)

    filtered.sort(key=sort_key, reverse=True)
    return filtered[:limit]


# ---------------------------------------------------------------------------
# HTML dump helper
# ---------------------------------------------------------------------------

def _build_poi_raw_html_dump(poi_li_blocks: List[str]) -> str:
    if not poi_li_blocks:
        return (
            "<!doctype html>\n"
            "<html><head><meta charset=\"utf-8\"><title>Booking POI LI dump</title></head>\n"
            "<body><h1>No POI li blocks found</h1></body></html>\n"
        )
    joined = "\n\n".join(poi_li_blocks)
    return (
        "<!doctype html>\n"
        "<html><head><meta charset=\"utf-8\"><title>Booking POI LI dump</title></head>\n"
        "<body><h1>Extracted li blocks</h1><ul>\n"
        f"{joined}\n"
        "</ul></body></html>\n"
    )


# ---------------------------------------------------------------------------
# Fetch pipeline (unchanged except attractions_url construction)
# ---------------------------------------------------------------------------

def resolve_destination(
    country: str,
    city: str,
    proxy_urls: Optional[List[str]] = None,
    cookie_header: str = "",
) -> dict:
    query = f"{city}, {country}" if country else city
    payload = {
        "query": query,
        "pageview_id": "",
        "aid": 800210,
        "language": "en-us",
        "size": 5,
    }
    attempts: List[str] = []
    candidates = [None] + (proxy_urls or [])
    data = None
    for proxy_url in candidates:
        try:
            data = _http_json(AUTOCOMPLETE_URL, payload=payload, proxy_url=proxy_url, cookie_header=cookie_header)
            break
        except Exception as exc:
            label = proxy_url or "direct"
            attempts.append(f"{label}: {exc}")

    if data is None:
        raise RuntimeError("Autocomplete request failed. " + " | ".join(attempts))

    results = data.get("results") or []
    if not results:
        raise RuntimeError(f"No Booking destination suggestions found for {query!r}")

    wanted_city = city.lower().strip()
    wanted_country = country.lower().strip()

    def score(item: dict) -> tuple:
        label = " ".join(
            str(item.get(field, "")) for field in ("label", "label1", "label2", "value")
        ).lower()
        cc1 = str(item.get("cc1", "")).lower()
        dest_type = str(item.get("dest_type", "")).lower()
        city_match = 1 if wanted_city and wanted_city in label else 0
        country_match = 1 if wanted_country and wanted_country in label else 0
        country_code_match = 1 if wanted_country and wanted_country == cc1 else 0
        destination_bonus = 1 if dest_type in {"city", "district", "region", "country"} else 0
        return (city_match + country_match + country_code_match + destination_bonus, len(label))

    return sorted(results, key=score, reverse=True)[0]


def build_attractions_start_url(city: str, country: str) -> str:
    query = f"{city}, {country}" if country else city
    params = {"ss": query, "lang": "en-gb"}
    return f"{ATTRACTIONS_LANDING_URL}?{urlencode(params)}"


def fetch_search_html(
    url: str,
    proxy_urls: Optional[List[str]] = None,
    cookie_header: str = "",
    cookie_records: Optional[List[dict]] = None,
    profile_dir: Optional[str] = None,
    profile_name: str = "Default",
    save_cookies_file: Optional[str] = None,
    prefer_browser: bool = False,
    manual_unblock: bool = False,
    debug_browser: bool = False,
    headless: bool = False,
    page_load_timeout: int = 45,
    content_timeout: int = 35,
    search_query: Optional[str] = None,
    require_attr_results: bool = False,
) -> str:
    attempts: List[str] = []
    candidates = [None] + (proxy_urls or [])
    blocked_html: Optional[str] = None

    if search_query:
        prefer_browser = True

    if not prefer_browser:
        for proxy_url in candidates:
            label = _proxy_label(proxy_url)
            try:
                html_text = _http_text(
                    url,
                    headers={"Referer": "https://www.booking.com/"},
                    proxy_url=proxy_url,
                    cookie_header=cookie_header,
                )
                if _looks_blocked(html_text):
                    if blocked_html is None:
                        blocked_html = html_text
                    attempts.append(f"{label}: bot-challenge")
                    continue
                if _looks_empty_or_invalid_html(html_text):
                    attempts.append(f"{label}: empty-or-invalid-html")
                    continue
                return html_text
            except Exception as exc:
                attempts.append(f"{label}: {exc}")

    browser_candidates = [p for p in (proxy_urls or []) if _proxy_supports_edge_proxy_arg(p)]
    browser_candidates = [None] + browser_candidates + [p for p in (proxy_urls or []) if p not in browser_candidates]

    for proxy_url in browser_candidates:
        label = f"edge/{_proxy_label(proxy_url)}"
        try:
            html_text = _fetch_with_browser(
                url,
                proxy_url=proxy_url,
                cookie_records=cookie_records or [],
                profile_dir=profile_dir,
                profile_name=profile_name,
                save_cookies_file=save_cookies_file,
                manual_unblock=manual_unblock,
                debug_browser=debug_browser,
                headless=headless,
                page_load_timeout=page_load_timeout,
                content_timeout=content_timeout,
                search_query=search_query,
                require_attr_results=require_attr_results,
            )
            return html_text
        except Exception as exc:
            attempts.append(f"{label}: {exc}")

    if blocked_html is not None:
        return blocked_html

    raise RuntimeError(
        "Booking fetch failed after trying direct + proxy HTTP and Edge fallback. "
        + " | ".join(attempts)
    )


def _proxy_supports_edge_proxy_arg(proxy_url: str) -> bool:
    split = urlsplit(proxy_url)
    return not split.username and not split.password


def _edge_proxy_server_arg(proxy_url: str) -> str:
    split = urlsplit(proxy_url)
    if split.username or split.password:
        raise RuntimeError("Edge proxy arg does not support authenticated proxy URLs")
    if not split.hostname or not split.port:
        raise RuntimeError("Invalid proxy URL")
    return f"http://{split.hostname}:{split.port}"


def _apply_cookies_to_driver(driver, cookie_records: List[dict]) -> None:
    if not cookie_records:
        return
    for cookie in cookie_records:
        name = str(cookie.get("name", "")).strip()
        value = str(cookie.get("value", ""))
        if not name:
            continue
        cookie_data = {"name": name, "value": value}
        for key in ("domain", "path", "expiry", "secure", "httpOnly", "sameSite"):
            if key in cookie and cookie[key] not in (None, ""):
                cookie_data[key] = cookie[key]
        try:
            driver.add_cookie(cookie_data)
        except Exception:
            continue


def _dump_driver_cookies(driver, cookie_file: Optional[str]) -> None:
    if not cookie_file:
        return
    try:
        driver.get_cookies()
    except Exception:
        return


def _debug_driver_state(driver, label: str) -> None:
    try:
        current_url = driver.current_url
    except Exception:
        current_url = "<unavailable>"
    try:
        title = driver.title
    except Exception:
        title = "<unavailable>"
    try:
        ready_state = driver.execute_script("return document.readyState")
    except Exception:
        ready_state = "<unavailable>"
    try:
        body_text = driver.find_element("tag name", "body").text or ""
    except Exception:
        body_text = ""
    try:
        page_source = driver.page_source or ""
    except Exception:
        page_source = ""
    print(f"[{label}] url={current_url}")
    print(f"[{label}] title={title!r}")
    print(f"[{label}] readyState={ready_state!r}")
    print(f"[{label}] body_chars={len(body_text)} page_chars={len(page_source)}")
    snippet = _normalize_whitespace(page_source[:500])
    if snippet:
        print(f"[{label}] page_snippet={snippet}")


def _save_browser_screenshot(driver, filename: str = "booking_debug.png") -> None:
    try:
        driver.save_screenshot(filename)
    except Exception:
        return


def _configure_driver_for_speed(driver) -> None:
    try:
        driver.execute_cdp_cmd("Network.enable", {})
        driver.execute_cdp_cmd(
            "Network.setBlockedURLs",
            {"urls": ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.gif", "*.svg", "data:image/*"]},
        )
    except Exception:
        pass


def _dismiss_booking_popup(driver) -> None:
    selectors = [
        "button[aria-label='Dismiss sign in information.']",
        "button[aria-label='Dismiss sign-in info.']",
        "button[aria-label*='Dismiss sign in']",
        "button[aria-label*='Dismiss sign-in']",
    ]
    try:
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait

        for selector in selectors:
            try:
                btn = WebDriverWait(driver, 2, poll_frequency=0.2).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                )
                driver.execute_script("arguments[0].click();", btn)
                return
            except Exception:
                continue
    except Exception:
        pass


def _wait_for_booking_content(driver, timeout: int = 35) -> None:
    try:
        from selenium.common.exceptions import TimeoutException
        from selenium.webdriver.support.ui import WebDriverWait
    except Exception:
        return

    def _is_ready(current_driver) -> bool:
        try:
            ready_state = current_driver.execute_script("return document.readyState")
        except Exception:
            ready_state = "loading"
        try:
            page_source = current_driver.page_source or ""
        except Exception:
            page_source = ""
        if ready_state != "complete":
            return False
        if _looks_blocked(page_source):
            return False
        if _looks_empty_or_invalid_html(page_source):
            return False
        lower = page_source.lower()
        body_text = ""
        try:
            body_text = current_driver.find_element("tag name", "body").text or ""
        except Exception:
            pass
        signals = ["property-card", 'data-testid="property-card"', "hotel_name",
                   "stay details", "search results", "css-zp7rdd", "ccd5b150e3"]
        if any(signal in lower for signal in signals):
            return True
        if len(body_text.strip()) > 1200:
            return True
        if len(page_source.strip()) > 5000:
            return True
        return False

    try:
        from selenium.common.exceptions import TimeoutException
        from selenium.webdriver.support.ui import WebDriverWait
        WebDriverWait(driver, timeout, poll_frequency=1).until(_is_ready)
    except Exception:
        pass


def _wait_for_attr_results(driver, timeout: int = 35) -> None:
    try:
        from selenium.common.exceptions import TimeoutException
        from selenium.webdriver.support.ui import WebDriverWait
    except Exception:
        return

    def _is_attr_ready(current_driver) -> bool:
        try:
            html_text = current_driver.page_source or ""
        except Exception:
            return False
        lowered = html_text.lower()
        if "attr-search-results-page-main-content" not in lowered:
            return False
        return "<li" in lowered

    try:
        from selenium.common.exceptions import TimeoutException
        from selenium.webdriver.support.ui import WebDriverWait
        WebDriverWait(driver, timeout, poll_frequency=1).until(_is_attr_ready)
    except Exception:
        pass


def _wait_for_attractions_results_page(driver, timeout: int = 25) -> bool:
    try:
        from selenium.common.exceptions import TimeoutException
        from selenium.webdriver.support.ui import WebDriverWait
    except Exception:
        return False

    def _has_results_page(current_driver) -> bool:
        try:
            current_url = (current_driver.current_url or "").lower()
        except Exception:
            current_url = ""
        if "/attractions/searchresults" in current_url:
            return True
        try:
            html_text = current_driver.page_source or ""
        except Exception:
            html_text = ""
        return "attr-search-results-page-main-content" in html_text.lower()

    try:
        from selenium.common.exceptions import TimeoutException
        from selenium.webdriver.support.ui import WebDriverWait
        WebDriverWait(driver, timeout, poll_frequency=0.5).until(_has_results_page)
        return True
    except Exception:
        return False


def _open_destination_card_from_landing(driver, city: str) -> bool:
    city_lower = city.strip().lower()
    if not city_lower:
        return False
    script = """
        const city = arguments[0].toLowerCase();
        const links = Array.from(document.querySelectorAll("a[href*='/attractions/searchresults']"));
        for (const link of links) {
            const aria = (link.getAttribute('aria-label') || '').toLowerCase();
            const title = (link.getAttribute('title') || '').toLowerCase();
            const text = (link.textContent || '').toLowerCase();
            if (
                aria.startsWith(city + ',') ||
                title.startsWith(city + ',') ||
                aria.includes(city) ||
                title.includes(city) ||
                text.includes(city)
            ) {
                link.click();
                return true;
            }
        }
        return false;
    """
    try:
        return bool(driver.execute_script(script, city_lower))
    except Exception:
        return False


def _submit_attractions_search(driver, search_query: str) -> bool:
    try:
        from selenium.webdriver.common.keys import Keys
    except Exception:
        return False

    field = None
    selectors = [
        "input[data-testid='search-input-field']",
        "input.SearchBoxFieldAutocomplete_input",
        "input[name='query']",
        "input[placeholder*='Where are you going' i]",
        "input[aria-label*='destination' i]",
    ]
    for selector in selectors:
        try:
            field = driver.find_element("css selector", selector)
            if field:
                break
        except Exception:
            continue

    if field is None:
        return False

    try:
        field.click()
    except Exception:
        pass
    try:
        field.send_keys(Keys.CONTROL, "a")
        field.send_keys(Keys.DELETE)
    except Exception:
        try:
            field.clear()
        except Exception:
            pass
    try:
        field.send_keys(search_query)
    except Exception:
        return False
    try:
        field.send_keys(Keys.ARROW_DOWN)
    except Exception:
        pass
    try:
        search_btn = driver.find_element("css selector", "button[data-testid='search-button']")
        driver.execute_script("arguments[0].click();", search_btn)
        return True
    except Exception:
        pass
    try:
        field.send_keys(Keys.ENTER)
        return True
    except Exception:
        pass
    try:
        return bool(
            driver.execute_script(
                """
                const input = arguments[0];
                const form = input.form || input.closest('form');
                if (!form) return false;
                if (typeof form.requestSubmit === 'function') {
                    form.requestSubmit();
                } else {
                    form.submit();
                }
                return true;
                """,
                field,
            )
        )
    except Exception:
        return False


def _fetch_with_browser(
    url: str,
    proxy_url: Optional[str] = None,
    cookie_records: Optional[List[dict]] = None,
    profile_dir: Optional[str] = None,
    profile_name: str = "Default",
    save_cookies_file: Optional[str] = None,
    manual_unblock: bool = False,
    debug_browser: bool = False,
    headless: bool = False,
    page_load_timeout: int = 45,
    content_timeout: int = 35,
    search_query: Optional[str] = None,
    require_attr_results: bool = False,
) -> str:
    try:
        from selenium import webdriver
        from selenium.common.exceptions import TimeoutException
        from selenium.webdriver.common.keys import Keys
        from selenium.webdriver.edge.options import Options as EdgeOptions
        from selenium.webdriver.edge.service import Service as EdgeService
        from selenium.webdriver.support.ui import WebDriverWait
    except Exception as exc:
        raise RuntimeError(
            "Booking blocked the HTTP request and selenium is not available. "
            "Install selenium, or run from a network that can reach Booking directly."
        ) from exc

    def _candidate_driver_path(env_keys: List[str]) -> Optional[str]:
        for key in env_keys:
            value = os.environ.get(key, "").strip().strip('"')
            if value and Path(value).exists():
                return value
        return None

    options = EdgeOptions()
    options.page_load_strategy = "eager"
    if headless and not manual_unblock:
        options.add_argument("--headless=new")
    prefs = {
        "profile.managed_default_content_settings.images": 2,
        "profile.managed_default_content_settings.media_stream": 2,
    }
    options.add_experimental_option("prefs", prefs)
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument(f"--user-agent={USER_AGENT}")
    if profile_dir:
        options.add_argument(f"--user-data-dir={profile_dir}")
        options.add_argument(f"--profile-directory={profile_name}")

    use_seleniumwire = False
    seleniumwire_webdriver = None
    seleniumwire_options = None
    if proxy_url:
        if _proxy_supports_edge_proxy_arg(proxy_url):
            options.add_argument(f"--proxy-server={_edge_proxy_server_arg(proxy_url)}")
        else:
            try:
                from seleniumwire import webdriver as seleniumwire_webdriver  # type: ignore
                seleniumwire_options = {
                    "proxy": {
                        "http": proxy_url,
                        "https": proxy_url,
                        "no_proxy": "localhost,127.0.0.1",
                    }
                }
                use_seleniumwire = True
            except Exception as exc:
                raise RuntimeError(
                    "Authenticated proxy requires selenium-wire for Edge. "
                    "Install it with: py -m pip install selenium-wire"
                ) from exc

    driver_path = _candidate_driver_path(["MSEDGEDRIVER", "WEBDRIVER_EDGE_DRIVER", "EDGEWEBDRIVER"])
    driver = None
    try:
        if use_seleniumwire:
            if driver_path:
                driver = seleniumwire_webdriver.Edge(  # type: ignore
                    options=options,
                    service=EdgeService(executable_path=driver_path),
                    seleniumwire_options=seleniumwire_options,
                )
            else:
                driver = seleniumwire_webdriver.Edge(  # type: ignore
                    options=options,
                    seleniumwire_options=seleniumwire_options,
                )
        elif driver_path:
            driver = webdriver.Edge(options=options, service=EdgeService(executable_path=driver_path))
        else:
            driver = webdriver.Edge(options=options)

        _configure_driver_for_speed(driver)
        driver.set_page_load_timeout(max(10, page_load_timeout))
        driver.set_script_timeout(max(10, page_load_timeout))

        if cookie_records:
            try:
                driver.get("https://www.booking.com/")
            except Exception:
                try:
                    driver.execute_script("window.stop();")
                except Exception:
                    pass
            _wait_for_booking_content(driver, timeout=min(12, content_timeout))
            _apply_cookies_to_driver(driver, cookie_records)

        try:
            driver.get(url)
        except Exception:
            try:
                driver.execute_script("window.stop();")
            except Exception:
                pass
            if debug_browser:
                print(f"[after-get] page load timed out after {max(10, page_load_timeout)}s; continuing with partial DOM")

        if search_query:
            submitted = _submit_attractions_search(driver, search_query)
            if not submitted:
                search_inputs = [
                    "input[name='ss']",
                    "input[type='search']",
                    "input[aria-label*='destination' i]",
                    "input[placeholder*='destination' i]",
                ]
                for selector in search_inputs:
                    try:
                        from selenium.webdriver.common.keys import Keys
                        field = driver.find_element("css selector", selector)
                        field.clear()
                        field.send_keys(search_query)
                        field.send_keys(Keys.ENTER)
                        submitted = True
                        break
                    except Exception:
                        continue
            if require_attr_results:
                transitioned = _wait_for_attractions_results_page(driver, timeout=max(8, content_timeout // 2))
                if not transitioned:
                    city_hint = search_query.split(",", 1)[0].strip()
                    if city_hint and _open_destination_card_from_landing(driver, city_hint):
                        _wait_for_attractions_results_page(driver, timeout=max(8, content_timeout // 2))
                _wait_for_attr_results(driver, timeout=content_timeout)

        _dismiss_booking_popup(driver)
        _dismiss_booking_popup(driver)
        if debug_browser:
            _debug_driver_state(driver, "after-get")
        _wait_for_booking_content(driver, timeout=content_timeout)
        _dismiss_booking_popup(driver)
        _wait_for_booking_content(driver, timeout=5)
        _dismiss_booking_popup(driver)
        html_text = driver.page_source
        if debug_browser:
            _debug_driver_state(driver, "after-wait")

        if manual_unblock:
            current_title = ""
            current_body_chars = 0
            try:
                current_title = (driver.title or "").strip()
            except Exception:
                pass
            try:
                current_body_chars = len((driver.find_element("tag name", "body").text or "").strip())
            except Exception:
                pass
            if _looks_blocked(html_text) and (not current_title or current_body_chars < 400):
                print("Bot challenge detected. Solve it in the visible browser, then press Enter to continue...")
                input()
                _wait_for_booking_content(driver, timeout=content_timeout)
                html_text = driver.page_source
                if debug_browser:
                    _debug_driver_state(driver, "after-manual")
        if manual_unblock and _looks_blocked(html_text):
            print("Still on challenge page. After solving it in browser, press Enter to retry once...")
            input()
            _wait_for_booking_content(driver, timeout=content_timeout)
            html_text = driver.page_source
            if debug_browser:
                _debug_driver_state(driver, "after-manual-retry")
        if debug_browser and _looks_blocked(html_text):
            _save_browser_screenshot(driver)

        _dump_driver_cookies(driver, save_cookies_file)
        return html_text
    finally:
        if driver is not None:
            driver.quit()


# ---------------------------------------------------------------------------
# Default dates
# ---------------------------------------------------------------------------

def default_dates() -> Tuple[str, str]:
    today = date.today()
    checkin = today + timedelta(days=7)
    checkout = checkin + timedelta(days=1)
    return checkin.isoformat(), checkout.isoformat()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape Booking.com Attractions and return rich structured POI data."
    )
    # ---- destination ----
    parser.add_argument("--country", required=True, help="Country name, e.g. France")
    parser.add_argument("--city", required=True, help="City name, e.g. Paris")

    # ---- dates ----
    parser.add_argument("--checkin", help="Check-in date YYYY-MM-DD")
    parser.add_argument("--checkout", help="Check-out date YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=1, help="Stay length when --checkout is omitted")

    # ---- people ----
    parser.add_argument("--adults", "--group-size", type=int, default=2,
                        dest="adults", help="Number of adults / group size")
    parser.add_argument("--rooms", type=int, default=1, help="Number of rooms")

    # ---- constraints ----
    parser.add_argument("--min-rating", type=float, default=None,
                        help="Only return attractions with rating ≥ this value (e.g. 4.0)")
    parser.add_argument("--max-price", type=float, default=None,
                        help="Only return attractions whose 'from' price ≤ this value")

    # ---- output ----
    parser.add_argument("--limit", type=int, default=25, help="Maximum attractions to return")
    parser.add_argument("--output", help="Write JSON output to this file")
    parser.add_argument("--raw-html", help="Save fetched HTML to this file")
    parser.add_argument(
        "--raw-poi-html",
        default="booking_pois_raw_li.html",
        help="Save extracted raw POI <li> HTML to this file",
    )

    # ---- proxy / auth ----
    parser.add_argument("--proxy-file", default=str(DEFAULT_PROXY_FILE),
                        help="Path to proxy list (host:port or host:port:user:pass)")
    parser.add_argument("--proxy", help="Single proxy to force")
    parser.add_argument("--no-proxy", action="store_true", help="Disable all proxies")
    parser.add_argument("--force-proxy-with-session", action="store_true")
    parser.add_argument("--cookies-file", default=str(DEFAULT_SESSION_COOKIE_FILE))
    parser.add_argument("--save-cookies", help="Write browser cookies here after fetch")

    # ---- browser ----
    parser.add_argument("--manual-unblock", action="store_true")
    parser.add_argument("--debug-browser", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--page-load-timeout", type=int, default=45)
    parser.add_argument("--content-timeout", type=int, default=35)
    parser.add_argument("--profile-dir", help="Edge user-data-dir for an existing session")
    parser.add_argument("--profile-name", default="Default")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    cookie_records = _load_cookie_records(args.cookies_file)
    cookie_header = _cookie_header(cookie_records)
    has_session_state = bool(args.profile_dir or cookie_records)

    if args.no_proxy:
        proxy_urls: List[str] = []
    elif args.proxy:
        proxy_value = _parse_proxy_line(args.proxy)
        if not proxy_value:
            raise SystemExit("Invalid --proxy format. Use host:port or host:port:user:pass")
        proxy_urls = [proxy_value]
    else:
        proxy_urls = load_proxies(args.proxy_file)
        if has_session_state and not args.force_proxy_with_session:
            proxy_urls = []

    if args.checkin and not args.checkout:
        checkout_dt = datetime.strptime(args.checkin, "%Y-%m-%d") + timedelta(days=max(1, args.days))
        checkin = args.checkin
        checkout = checkout_dt.date().isoformat()
    elif not args.checkin and not args.checkout:
        checkin, checkout = default_dates()
    elif args.checkin and args.checkout:
        checkin, checkout = args.checkin, args.checkout
    else:
        raise SystemExit("Provide both --checkin and --checkout, or neither.")

    search_query = f"{args.city}, {args.country}" if args.country else args.city
    attractions_start_url = f"{ATTRACTIONS_LANDING_URL}?{urlencode({'lang': 'en-gb'})}"
    search_url = attractions_start_url

    html_text = fetch_search_html(
        search_url,
        proxy_urls=proxy_urls,
        cookie_header=cookie_header,
        cookie_records=cookie_records,
        profile_dir=args.profile_dir,
        profile_name=args.profile_name,
        save_cookies_file=args.save_cookies,
        prefer_browser=has_session_state,
        manual_unblock=args.manual_unblock,
        debug_browser=args.debug_browser,
        headless=args.headless,
        page_load_timeout=args.page_load_timeout,
        content_timeout=args.content_timeout,
        search_query=search_query,
        require_attr_results=True,
    )

    if args.raw_html:
        Path(args.raw_html).write_text(html_text, encoding="utf-8")

    if args.raw_poi_html:
        poi_li_blocks = _extract_poi_li_html(html_text)
        Path(args.raw_poi_html).write_text(_build_poi_raw_html_dump(poi_li_blocks), encoding="utf-8")

    attractions = extract_attractions(
        html_text,
        city=args.city,
        country=args.country,
        limit=args.limit,
        min_rating=args.min_rating,
        max_price=args.max_price,
    )

    result = {
        "query": {
            "city": args.city,
            "country": args.country,
            "checkin": checkin,
            "checkout": checkout,
            "adults": args.adults,
            "rooms": args.rooms,
            "constraints": {
                "min_rating": args.min_rating,
                "max_price": args.max_price,
            },
        },
        "search_url": search_url,
        "total_found": len(attractions),
        "attractions": attractions,
    }

    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

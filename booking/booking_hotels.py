"""Scrape Booking.com Hotels for a given city and date range.

Extracts rich structured data from each hotel card:
  name, url, location, description, rating, review_count, rating_label,
  price_per_night, currency, availability, image_alt, image_url.

Supports optional constraints:
  --max-price        Filter by maximum price per night (in page currency)
  --min-rating       Filter by minimum rating (e.g. 4.0)
  --rooms            Number of rooms
  --adults           Number of adults per room

Usage examples:
    python booking_hotels.py --country France --city Paris
    python booking_hotels.py --country France --city Paris --checkin 2026-05-01 --checkout 2026-05-02
    python booking_hotels.py --country UAE --city Dubai --max-price 150 --min-rating 4.0
    python booking_hotels.py --country "United Arab Emirates" --city "Abu Dhabi" --output hotels.json
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
BOOKING_BASE = "https://www.booking.com"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
DEFAULT_PROXY_FILE = Path(__file__).resolve().parents[1] / "scrape" / "Webshare 10 proxies.txt"
DEFAULT_SESSION_COOKIE_FILE = Path(__file__).resolve().parent / "booking_cookies.json"


# ---------------------------------------------------------------------------
# Proxy / cookie helpers
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
# Rich card extraction for hotels
# ---------------------------------------------------------------------------

def _parse_price(raw: str) -> Tuple[Optional[float], Optional[str]]:
    """Return (amount, currency_symbol) or (None, None) if unparseable."""
    cleaned = unescape(raw).replace("\xa0", " ").strip()
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


def _extract_hotel_cards(html_text: str) -> List[str]:
    """Return raw HTML of each hotel property card <div> block (deduplicated)."""
    hotel_blocks: List[str] = []
    seen: set = set()

    # Look for property cards - common patterns in Booking.com
    patterns = [
        r'<div\b[^>]*data-testid="[^"]*property-card[^"]*"[^>]*>[\s\S]*?</div>',
        r'<div\b[^>]*class="[^"]*propertycard[^"]*"[^>]*>[\s\S]*?</div>',
        r'<article\b[^>]*>[\s\S]*?</article>',
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, html_text, flags=re.I):
            block = match.group(0).strip()
            if not block:
                continue
            # Avoid sponsored/ads
            if re.search(r'\b(?:sponsored|advertisement|ad)\b', block, flags=re.I):
                continue
            key = re.sub(r"\s+", " ", block).strip().lower()
            if key in seen:
                continue
            seen.add(key)
            hotel_blocks.append(block)

    return hotel_blocks


def _parse_hotel_card(block: str) -> Optional[dict]:
    """
    Parse a single hotel product card <div> block into a rich dict.

    Returns None if the card has no recognisable title.
    """
    # ---- title & URL -------------------------------------------------------
    title_match = re.search(
        r'<h2\b[^>]*>[\s\S]*?<a\b[^>]*href=["\']([^"\']+)["\'][^>]*>([^<]+)</a>',
        block, flags=re.I,
    )
    if not title_match:
        title_match = re.search(
            r'<a\b[^>]*data-testid=["\']property-card-[^"\']*["\'][^>]*href=["\']([^"\']+)["\'][^>]*>[\s\S]*?<h2\b[^>]*>([^<]+)',
            block, flags=re.I,
        )
    if not title_match:
        title_match = re.search(
            r'<a\b[^>]*href=["\']([^"\']+)["\'][^>]*>[\s\S]*?<span\b[^>]*>([^<]+)</span>',
            block, flags=re.I,
        )

    if not title_match:
        return None

    raw_href = unescape(title_match.group(1))
    name = _strip_tags(title_match.group(2))

    if not name:
        return None

    url = BOOKING_BASE + raw_href if raw_href.startswith("/") else raw_href

    # ---- location ----------------------------------------------------------
    location = None
    location_match = re.search(r'<p\b[^>]*class="[^"]*mapmeta[^"]*"[^>]*>([^<]+)', block, re.I)
    if location_match:
        location = unescape(location_match.group(1).strip())
    else:
        location_match = re.search(r'<p\b[^>]*>([^<]*(?:km|mi|m) (?:from|to)[^<]*)</p>', block, re.I)
        if location_match:
            location = unescape(location_match.group(1).strip())

    # ---- description -------------------------------------------------------
    description = None
    desc_match = re.search(r'<p\b[^>]*class="[^"]*description[^"]*"[^>]*>([\s\S]*?)</p>', block, re.I)
    if desc_match:
        description = _strip_tags(desc_match.group(1))
    else:
        # Try generic p tags (skip location ones)
        for match in re.finditer(r'<p\b[^>]*>([^<]+)</p>', block, re.I):
            text = _strip_tags(match.group(1))
            if text and "km" not in text.lower() and "mi" not in text.lower():
                description = text
                break

    # ---- rating & review count ---------------------------------------------
    rating: Optional[float] = None
    review_count: Optional[int] = None
    rating_label: Optional[str] = None

    # Look for rating patterns
    rating_match = re.search(
        r'(?:score|rating)[\'"\s]*[=:]\s*([\d.]+)|(\d\.\d)\s*(?:out of|/)\s*\d',
        block, flags=re.I,
    )
    if rating_match:
        try:
            rating = float(rating_match.group(1) or rating_match.group(2))
        except (ValueError, TypeError):
            pass

    # Look for review count
    review_match = re.search(r'(\d+)\s*reviews?|reviews?:\s*(\d+)', block, flags=re.I)
    if review_match:
        try:
            review_count = int(review_match.group(1) or review_match.group(2))
        except (ValueError, TypeError):
            pass

    # Rating label (Fabulous, Superb, etc)
    label_match = re.search(r'(?:Fabulous|Superb|Very good|Good|Pleasant|OK|Average|Poor)[\'"\s]*</|>([A-Za-z]+)\s*</(?:span|p|div)', block, re.I)
    if label_match:
        rating_label = label_match.group(1).strip()

    # ---- price per night ---------------------------------------------------
    price_per_night: Optional[float] = None
    currency: Optional[str] = None

    # Look for price patterns
    price_match = re.search(
        r'(?:price|from|from price|starting)[\'"\s]*[=:]*\s*([^<\n]{1,20}(?:[\d,\.]+))',
        block, flags=re.I,
    )
    if price_match:
        price_per_night, currency = _parse_price(price_match.group(1))

    # Fallback: look for currency symbol followed by number
    if not price_per_night:
        price_fallback = re.search(r'([^\d\s]{1,3})\s*([\d,\.]+)', block)
        if price_fallback:
            price_per_night, currency = _parse_price(price_fallback.group(0))

    # ---- availability ------------------------------------------------------
    availability = None
    avail_match = re.search(r'(?:available|availability|available now)', block, flags=re.I)
    if avail_match:
        availability = "Available"
    else:
        avail_sold = re.search(r'(?:sold out|fully booked|no availability)', block, flags=re.I)
        if avail_sold:
            availability = "Not Available"

    # ---- image -------------------------------------------------------------
    image_alt = None
    image_url = None
    img_match = re.search(r'<img\b[^>]*(?:alt="([^"]*)"|src="([^"]+)")[^>]*>', block, re.I)
    if not img_match:
        img_match = re.search(r'<img\b[^>]*src="([^"]+)"[^>]*alt="([^"]*)"', block, re.I)
    if img_match:
        image_alt = img_match.group(1)
        image_url = img_match.group(2)

    return {
        "name": name,
        "url": url,
        "location": location,
        "description": description,
        "rating": rating,
        "rating_label": rating_label,
        "review_count": review_count,
        "price_per_night": price_per_night,
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
            if card["price_per_night"] is None or card["price_per_night"] > max_price:
                continue
        result.append(card)
    return result


# ---------------------------------------------------------------------------
# Main extraction entry point
# ---------------------------------------------------------------------------

def extract_hotels(
    html_text: str,
    city: str,
    country: str,
    limit: int = 50,
    min_rating: Optional[float] = None,
    max_price: Optional[float] = None,
) -> List[dict]:
    """
    Parse all hotel product cards from the page and return rich dicts.

    Applies optional constraints (min_rating, max_price) before returning.
    Results are sorted by rating descending (unrated last).
    """
    raw_cards: List[dict] = []
    seen_names: set = set()

    for block in _extract_hotel_cards(html_text):
        card = _parse_hotel_card(block)
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
# Fetch pipeline
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


def build_search_url(
    city: str,
    country: str,
    checkin: str,
    checkout: str,
    adults: int = 2,
    rooms: int = 1,
) -> str:
    """Build the search URL for hotels."""
    query = f"{city}, {country}" if country else city
    params = {
        "ss": query,
        "checkin": checkin,
        "checkout": checkout,
        "ssne": query,
        "ssne_untouched": query,
        "group_adults": adults,
        "no_rooms": rooms,
        "lang": "en-gb",
    }
    return f"{SEARCH_URL}?{urlencode(params)}"


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
) -> str:
    """Fetch hotel search results HTML."""
    attempts: List[str] = []
    candidates = [None] + (proxy_urls or [])
    blocked_html: Optional[str] = None

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

    # Try browser fallback if HTTP fails
    try:
        from selenium import webdriver
        from selenium.webdriver.edge.options import Options as EdgeOptions
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC

        options = EdgeOptions()
        options.page_load_strategy = "eager"
        options.add_argument("--headless=new") if headless else None
        options.add_argument("--no-sandbox")
        options.add_argument(f"--user-agent={USER_AGENT}")

        driver = webdriver.Edge(options=options)
        try:
            driver.get(url)
            WebDriverWait(driver, content_timeout).until(
                EC.presence_of_all_elements_located((By.CLASS_NAME, "propertycard"))
            )
            html_text = driver.page_source
            return html_text
        finally:
            driver.quit()
    except Exception as exc:
        attempts.append(f"browser: {exc}")

    if blocked_html is not None:
        return blocked_html

    raise RuntimeError(
        "Hotel fetch failed after trying direct + proxy HTTP and browser fallback. "
        + " | ".join(attempts)
    )


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
        description="Scrape Booking.com Hotels and return rich structured data."
    )
    # ---- destination ----
    parser.add_argument("--country", required=True, help="Country name, e.g. France")
    parser.add_argument("--city", required=True, help="City name, e.g. Paris")

    # ---- dates ----
    parser.add_argument("--checkin", help="Check-in date YYYY-MM-DD")
    parser.add_argument("--checkout", help="Check-out date YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=1, help="Stay length when --checkout is omitted")

    # ---- people ----
    parser.add_argument("--adults", type=int, default=2, help="Number of adults per room")
    parser.add_argument("--rooms", type=int, default=1, help="Number of rooms")

    # ---- constraints ----
    parser.add_argument("--min-rating", type=float, default=None,
                        help="Only return hotels with rating ≥ this value (e.g. 4.0)")
    parser.add_argument("--max-price", type=float, default=None,
                        help="Only return hotels whose price per night ≤ this value")

    # ---- output ----
    parser.add_argument("--limit", type=int, default=25, help="Maximum hotels to return")
    parser.add_argument("--output", help="Write JSON output to this file")
    parser.add_argument("--raw-html", help="Save fetched HTML to this file")

    # ---- proxy / auth ----
    parser.add_argument("--proxy-file", default=str(DEFAULT_PROXY_FILE),
                        help="Path to proxy list (host:port or host:port:user:pass)")
    parser.add_argument("--proxy", help="Single proxy to force")
    parser.add_argument("--no-proxy", action="store_true", help="Disable all proxies")
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
        if has_session_state:
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

    search_url = build_search_url(
        city=args.city,
        country=args.country,
        checkin=checkin,
        checkout=checkout,
        adults=args.adults,
        rooms=args.rooms,
    )

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
    )

    if args.raw_html:
        Path(args.raw_html).write_text(html_text, encoding="utf-8")
        print(f"[*] Raw HTML saved to {args.raw_html}")

    hotels = extract_hotels(
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
        "total_found": len(hotels),
        "hotels": hotels,
    }

    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"[*] Results saved to {args.output}")
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

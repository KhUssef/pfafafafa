"""Scrape Booking.com for POIs near a city search.

This script uses Booking.com's public autocomplete endpoint to resolve the
destination, then loads the public search results page and extracts nearby
place names from the listing descriptions.

It does not use any paid scraping API.

Usage examples:
    python booking_pois.py --country France --city Paris
    python booking_pois.py --country France --city Paris --checkin 2026-05-01 --checkout 2026-05-02
    python booking_pois.py --country "United Arab Emirates" --city Abu Dhabi --output pois.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta
from html import unescape
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import ProxyHandler, Request, build_opener, urlopen


AUTOCOMPLETE_URL = "https://accommodations.booking.com/autocomplete.json"
SEARCH_URL = "https://www.booking.com/searchresults.en-gb.html"
ATTRACTIONS_LANDING_URL = "https://www.booking.com/attractions/index.en-gb.html"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
DEFAULT_PROXY_FILE = Path(__file__).resolve().parents[1] / "scrape" / "Webshare 10 proxies.txt"
DEFAULT_SESSION_COOKIE_FILE = Path(__file__).resolve().parent / "booking_cookies.json"


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


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _strip_tags(html_text: str) -> str:
    text = re.sub(r"<script\b.*?</script>", " ", html_text, flags=re.S | re.I)
    text = re.sub(r"<style\b.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return _normalize_whitespace(unescape(text))


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

    def score(item: dict) -> tuple[int, int]:
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


def build_search_url(destination: dict, checkin: str, checkout: str, adults: int, rooms: int) -> str:
    params = {
        "ss": destination.get("value") or destination.get("label") or "",
        "ssne": destination.get("value") or destination.get("label") or "",
        "ssne_untouched": destination.get("value") or destination.get("label") or "",
        "checkin": checkin,
        "checkout": checkout,
        "no_rooms": rooms,
        "group_adults": adults,
        "group_children": 0,
        "dest_id": destination.get("dest_id", ""),
        "dest_type": destination.get("dest_type", ""),
        "lang": "en-gb",
        "sb": 1,
        "sb_travel_purpose": "leisure",
        "src": "index",
        "src_elem": "sb",
        "efdco": 1,
    }
    return f"{SEARCH_URL}?{urlencode(params)}"


def build_attractions_start_url(city: str, country: str) -> str:
    query = f"{city}, {country}" if country else city
    params = {
        "ss": query,
        "lang": "en-gb",
    }
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
                    headers={
                        "Referer": "https://www.booking.com/",
                    },
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
        cookies = driver.get_cookies()
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
                driver = seleniumwire_webdriver.Edge(  # type: ignore[union-attr]
                    options=options,
                    service=EdgeService(executable_path=driver_path),
                    seleniumwire_options=seleniumwire_options,
                )
            else:
                driver = seleniumwire_webdriver.Edge(  # type: ignore[union-attr]
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
            except TimeoutException:
                try:
                    driver.execute_script("window.stop();")
                except Exception:
                    pass
            _wait_for_booking_content(driver, timeout=min(12, content_timeout))
            _apply_cookies_to_driver(driver, cookie_records)
        try:
            driver.get(url)
        except TimeoutException:
            # Recover from partial page hangs instead of getting stuck on navigation.
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
                current_title = ""
            try:
                current_body_chars = len((driver.find_element("tag name", "body").text or "").strip())
            except Exception:
                current_body_chars = 0

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


def _configure_driver_for_speed(driver) -> None:
    try:
        driver.execute_cdp_cmd("Network.enable", {})
        driver.execute_cdp_cmd(
            "Network.setBlockedURLs",
            {
                "urls": [
                    "*.png",
                    "*.jpg",
                    "*.jpeg",
                    "*.webp",
                    "*.gif",
                    "*.svg",
                    "data:image/*",
                ]
            },
        )
    except Exception:
        pass


def _dismiss_booking_popup(driver):
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait
    
    # More comprehensive popup selectors
    selectors = [
        # Sign-in popups
        "button[aria-label='Dismiss sign in information.']",
        "button[aria-label='Dismiss sign-in info.']",
        # Cookie consent
        "button[id='onetrust-accept-btn-handler']",
        "button[aria-label='Cookie settings']",
        "button:contains('Accept All Cookies')",
        # Newsletter/promotional modals
        "button[aria-label='Close']",
        "button[data-modal-close='true']",
        "div[role='dialog'] button[aria-label='Close']",
        # Genius popups
        "button[aria-label='Dismiss']",
        # Generic close buttons
        "button[class*='close']",
        "div[class*='modal'] button[type='button']",
    ]
    
    try:
        # Wait for any popup to appear (but don't wait too long)
        for selector in selectors:
            try:
                btn = WebDriverWait(driver, 1, poll_frequency=0.2).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                )
                driver.execute_script("arguments[0].click();", btn)
                time.sleep(0.5)  # Give time for popup animation
                return True
            except Exception:
                continue
    except Exception:
        pass
    return False


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

        signals = [
            "property-card",
            "data-testid=\"property-card\"",
            "hotel_name",
            "stay details",
            "search results",
            "css-zp7rdd",
            "ccd5b150e3",
        ]
        if any(signal in lower for signal in signals):
            return True
        if len(body_text.strip()) > 1200:
            return True
        if len(page_source.strip()) > 5000:
            return True
        return False

    try:
        WebDriverWait(driver, timeout, poll_frequency=1).until(_is_ready)
    except TimeoutException:
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
        WebDriverWait(driver, timeout, poll_frequency=1).until(_is_attr_ready)
    except TimeoutException:
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
        WebDriverWait(driver, timeout, poll_frequency=0.5).until(_has_results_page)
        return True
    except TimeoutException:
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

    # Booking often requires keyboard interaction to select the autocomplete option.
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


def _extract_paragraph_texts(html_text: str) -> List[str]:
    paragraphs = []
    for match in re.finditer(r"<p\b[^>]*>(.*?)</p>", html_text, flags=re.S | re.I):
        text = _strip_tags(match.group(1))
        if text:
            paragraphs.append(text)
    return paragraphs


def _extract_landmarks_from_html(html_text: str) -> List[str]:
    landmarks: List[str] = []
    seen = set()

    # Sidebar filters expose nearby landmarks under the popular_nearby_landmarks group.
    for match in re.finditer(
        r'data-filters-item="popular_nearby_landmarks:[^"]*"[\s\S]{0,240}?aria-label="([^"]+?):\s*\d+\s+properties"',
        html_text,
        flags=re.I,
    ):
        candidate = _normalize_whitespace(unescape(match.group(1)))
        if candidate and candidate.lower() not in seen:
            seen.add(candidate.lower())
            landmarks.append(candidate)

    # Apollo payload keeps the same filter values in JSON form.
    for match in re.finditer(
        r'"urlId":"popular_nearby_landmarks=\d+"[\s\S]{0,260}?"text":"([^"]+)"',
        html_text,
        flags=re.I,
    ):
        candidate = _normalize_whitespace(unescape(match.group(1).replace("\\u0026", "&")))
        if candidate and candidate.lower() not in seen:
            seen.add(candidate.lower())
            landmarks.append(candidate)

    return landmarks


def _extract_property_titles(html_text: str) -> List[str]:
    titles: List[str] = []
    seen = set()
    for match in re.finditer(r'data-testid="title"[^>]*>(.*?)</div>', html_text, flags=re.S | re.I):
        title = _strip_tags(match.group(1))
        if not title:
            continue
        key = title.lower()
        if key in seen:
            continue
        seen.add(key)
        titles.append(title)
    return titles


def _extract_attr_results_sections(html_text: str) -> List[str]:
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
    poi_blocks: List[str] = []
    seen = set()

    for section_html in _extract_attr_results_sections(html_text):
        for match in re.finditer(r"<li\b[^>]*>[\s\S]*?</li>", section_html, flags=re.I):
            block = match.group(0).strip()
            if not block:
                continue
            # Ignore non-result banners/carousels and keep only product-card rows.
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


def _extract_pois_from_booking_cards(html_text: str, city: str, country: str) -> List[str]:
    pois: List[str] = []
    seen = set()

    for raw_li_html in _extract_poi_li_html(html_text):
        title_match = re.search(
            r'<h3\b[^>]*data-testid=["\']card-title["\'][^>]*>(.*?)</h3>',
            raw_li_html,
            flags=re.S | re.I,
        )
        if not title_match:
            title_match = re.search(
                r'<a\b[^>]*href=["\'][^"\']*/attractions/[^"\']*["\'][^>]*>(.*?)</a>',
                raw_li_html,
                flags=re.S | re.I,
            )

        if not title_match:
            continue

        candidate = _clean_candidate(_strip_tags(title_match.group(1)), city, country)
        if not candidate:
            continue

        key = candidate.casefold()
        if key in seen:
            continue
        seen.add(key)
        pois.append(candidate)

    return pois


def _extract_pois_from_title(title: str, city: str, country: str) -> List[str]:
    cues = [
        "walk to",
        "near",
        "next to",
        "close to",
        "opposite",
        "across from",
        "facing",
        "views of",
        "view of",
    ]
    lower = title.lower()
    found: List[str] = []

    for cue in cues:
        start = lower.find(cue)
        if start == -1:
            continue

        tail = title[start + len(cue) :]
        tail = re.split(r"\b(?:by|in|at|with)\b", tail, maxsplit=1, flags=re.I)[0]
        for part in re.split(r"\s*(?:,|/|\\+| and | & )\s*", tail, flags=re.I):
            candidate = _clean_candidate(part, city, country)
            if candidate:
                found.append(candidate)

    return found


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

    blocked_exact = {
        city.lower(),
        country.lower(),
        f"{city}, {country}".lower(),
    }
    if cleaned.lower() in blocked_exact:
        return None
    blocked_fragments = (
        "make an informed decision",
        "highly rated",
        "featured",
        "value for money",
        "show on map",
        "opens in new window",
        "properties found",
        "reviews",
        "free cancellation",
        "breakfast",
        "private bathroom",
        "air conditioning",
        "swimming pool",
        "hot tub",
        "kitchen",
        "parking",
        "stars",
        "hotel",
        "guest house",
        "holiday home",
    )
    lowered = cleaned.lower()
    if any(fragment in lowered for fragment in blocked_fragments):
        return None
    if lowered in {"downtown", "downtown dubai", "dubai"}:
        return None
    if len(cleaned) < 3:
        return None
    return cleaned


def _extract_pois_from_text(text: str, city: str, country: str) -> List[str]:
    patterns = [
        r"(?:from|near|nearby|to|with|within|close to|next to|opposite)\s+(?:the\s+)?([A-Z][^.,;()]{2,120})",
        r"(?:miles?|kilomet(?:er|re)s?|km|meters?|metres?|minutes?)\s+(?:from|to|of)?\s+(?:the\s+)?([A-Z][^.,;()]{2,120})",
        r"([A-Z][A-Za-z0-9'&()./-]*(?:\s+[A-Z][A-Za-z0-9'&()./-]*){1,5})\s+(?:reachable within|is within|within)\b",
    ]

    found = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            candidate = _clean_candidate(match.group(1), city, country)
            if candidate:
                found.append(candidate)

    return found


def extract_pois(html_text: str, city: str, country: str, limit: int = 50) -> List[dict]:
    mentions: Dict[str, List[str]] = defaultdict(list)
    counts: Counter = Counter()
    display_name: Dict[str, str] = {}

    def add_candidate(name: str, source_text: str, weight: int) -> None:
        key = name.casefold()
        counts[key] += weight
        if key not in display_name:
            display_name[key] = name
        if source_text not in mentions[key]:
            mentions[key].append(source_text)

    # Primary source: POI text blocks rendered in the search results cards.
    for card_poi in _extract_pois_from_booking_cards(html_text, city, country):
        add_candidate(card_poi, "booking_card_poi", 6)

    # Secondary source: Booking's own nearby landmarks filters.
    for landmark in _extract_landmarks_from_html(html_text):
        candidate = _clean_candidate(landmark, city, country)
        if not candidate:
            continue
        marker = "popular_nearby_landmarks"
        add_candidate(candidate, marker, 5)

    # Tertiary source: property-card titles with proximity phrasing.
    for title in _extract_property_titles(html_text):
        candidates = _extract_pois_from_title(title, city, country)
        if not candidates:
            continue
        for candidate in candidates:
            add_candidate(candidate, title, 1)

    # Fallback source: paragraphs, only when structured sources are empty.
    if not counts:
        for paragraph in _extract_paragraph_texts(html_text):
            if not re.search(r"\b(?:mile|km|meter|metre|minute|walk|reach|close|near|from|to)\b", paragraph, flags=re.I):
                continue
            candidates = _extract_pois_from_text(paragraph, city, country)
            if not candidates:
                continue
            for candidate in candidates:
                add_candidate(candidate, paragraph, 1)

    pois = []
    for key, count in counts.most_common(limit):
        pois.append(
            {
                "name": display_name[key],
                "mentions": count,
                "examples": mentions[key][:3],
            }
        )
    return pois


def default_dates() -> tuple[str, str]:
    today = date.today()
    checkin = today + timedelta(days=7)
    checkout = checkin + timedelta(days=1)
    return checkin.isoformat(), checkout.isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape Booking.com POIs for a city search.")
    parser.add_argument("--country", required=True, help="Country name, for example France")
    parser.add_argument("--city", required=True, help="City name, for example Paris")
    parser.add_argument("--checkin", help="Check-in date in YYYY-MM-DD format")
    parser.add_argument("--checkout", help="Check-out date in YYYY-MM-DD format")
    parser.add_argument("--days", type=int, default=1, help="Length of stay when checkout is omitted")
    parser.add_argument("--adults", type=int, default=2, help="Number of adults")
    parser.add_argument("--rooms", type=int, default=1, help="Number of rooms")
    parser.add_argument("--limit", type=int, default=25, help="Maximum number of POIs to return")
    parser.add_argument("--output", help="Write the JSON output to a file")
    parser.add_argument("--raw-html", help="Optional path to save the fetched HTML")
    parser.add_argument(
        "--raw-poi-html",
        default="booking_pois_raw_li.html",
        help="Path to save extracted raw POI li HTML blocks for inspection",
    )
    parser.add_argument(
        "--proxy-file",
        default=str(DEFAULT_PROXY_FILE),
        help="Path to a proxy list file (host:port or host:port:user:pass)",
    )
    parser.add_argument("--proxy", help="Single proxy to force, same format as proxy file lines")
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="Disable proxies and use direct traffic only",
    )
    parser.add_argument(
        "--force-proxy-with-session",
        action="store_true",
        help="Keep proxy rotation even when profile/cookies are provided",
    )
    parser.add_argument(
        "--cookies-file",
        default=str(DEFAULT_SESSION_COOKIE_FILE),
        help="Path to cookies JSON/Netscape file used for both HTTP and browser sessions",
    )
    parser.add_argument(
        "--save-cookies",
        help="Write cookies from a successful browser session to this JSON file",
    )
    parser.add_argument(
        "--manual-unblock",
        action="store_true",
        help="Open visible browser, let you solve challenge manually, then continue",
    )
    parser.add_argument(
        "--debug-browser",
        action="store_true",
        help="Print browser state and save a screenshot when the page does not load correctly",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Use headless browser mode (default is headful)",
    )
    parser.add_argument(
        "--page-load-timeout",
        type=int,
        default=45,
        help="Maximum seconds for browser navigation before forcing a partial-page fallback",
    )
    parser.add_argument(
        "--content-timeout",
        type=int,
        default=35,
        help="Maximum seconds to wait for Booking content markers after navigation",
    )
    parser.add_argument(
        "--profile-dir",
        help="Edge user-data-dir to reuse an existing browser session and its cookies",
    )
    parser.add_argument("--profile-name", default="Default", help="Edge profile directory name")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cookie_records = _load_cookie_records(args.cookies_file)
    cookie_header = _cookie_header(cookie_records)
    has_session_state = bool(args.profile_dir or cookie_records)

    if args.no_proxy:
        proxy_urls = []
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
        checkout = datetime.strptime(args.checkin, "%Y-%m-%d") + timedelta(days=max(1, args.days))
        checkin = args.checkin
        checkout = checkout.date().isoformat()
    elif not args.checkin and not args.checkout:
        checkin, checkout = default_dates()
    elif args.checkin and args.checkout:
        checkin, checkout = args.checkin, args.checkout
    else:
        raise SystemExit("Provide both --checkin and --checkout, or neither.")

    destination = {
        "label": f"{args.city}, {args.country}" if args.country else args.city,
        "value": f"{args.city}, {args.country}" if args.country else args.city,
        "dest_id": None,
        "dest_type": "attractions-search",
        "cc1": None,
    }
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

    pois = extract_pois(html_text, args.city, args.country, limit=args.limit)
    result = {
        "query": {
            "city": args.city,
            "country": args.country,
            "checkin": checkin,
            "checkout": checkout,
            "adults": args.adults,
            "rooms": args.rooms,
        },
        "destination": {
            "label": destination.get("label"),
            "value": destination.get("value"),
            "dest_id": destination.get("dest_id"),
            "dest_type": destination.get("dest_type"),
            "cc1": destination.get("cc1"),
        },
        "search_url": search_url,
        "attractions_start_url": attractions_start_url,
        "pois": pois,
    }

    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
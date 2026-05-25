"""Selenium-based Booking search script (uses selenium-wire for proxy support).

Usage:
    python booking_search.py --ss "Berlin, Germany" --proxy 31.59.20.176:6754

Proxy formats supported:
    host:port
    host:port:user:pass

Saves rendered HTML to `booking.html` and (optionally) prettified `booking_parsed.html` if bs4 is installed.
"""
import argparse
import time
import random
import sys
from urllib.parse import urlparse, parse_qs, urlencode
import re
import json

try:
    from seleniumwire import webdriver
    from selenium.webdriver.edge.options import Options
except Exception as e:
    print('Selenium or selenium-wire not installed:', e)
    print('Install with: python -m pip install selenium selenium-wire')
    sys.exit(1)


# public Booking search results base URL
BASE_SEARCH_URL = 'https://www.booking.com/searchresults.html'


def build_search_url(base_search_url: str, ss: str) -> str:
    # base_search_url is expected like 'https://www.booking.com/searchresults.html'
    p = urlparse(base_search_url)
    # keep default search params similar to the site; the function expects
    # the base_search_url to already contain baseline query params if needed.
    qs = parse_qs(p.query)
    qs['ss'] = [ss]
    return f"{p.scheme}://{p.netloc}{p.path}?{urlencode(qs, doseq=True)}"


def parse_proxy(proxy_str: str):
    # returns seleniumwire_options dict or None
    if not proxy_str:
        return None
    parts = proxy_str.split(':')
    if len(parts) == 2:
        host, port = parts
        return {'proxy': {'http': f'http://{host}:{port}', 'https': f'https://{host}:{port}', 'no_proxy': 'localhost,127.0.0.1'}}
    elif len(parts) >= 4:
        # host:port:user:pass (user or pass may contain colons, so split max 3)
        host = parts[0]
        port = parts[1]
        user = parts[2]
        pwd = ':'.join(parts[3:])
        auth = f"{user}:{pwd}@{host}:{port}"
        return {'proxy': {'http': f'http://{auth}', 'https': f'https://{auth}', 'no_proxy': 'localhost,127.0.0.1'}}
    else:
        return None


def do_search_selenium(base_search_url: str, ss: str, proxy: str = None, headless: bool = True):
    seleniumwire_options = parse_proxy(proxy)
    if seleniumwire_options:
        # show only host:port to avoid printing credentials
        p = proxy.split(':')
        print('Using proxy:', f"{p[0]}:{p[1]}")

    options = Options()
    if headless:
        options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('window-size=1920,1080')
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
    # be permissive about certs if proxy interferes
    options.add_argument('--ignore-certificate-errors')

    url = build_search_url(base_search_url, ss)
    print('Opening:', url)

    driver = None
    try:
        if seleniumwire_options:
            driver = webdriver.Edge(seleniumwire_options=seleniumwire_options, options=options)
        else:
            driver = webdriver.Edge(options=options)
        driver.get(url)
        # allow JS to run; increase if page needs more time
        time.sleep(6)
        page_html = driver.page_source
        print('Title:', driver.title)
        with open('booking.html', 'w', encoding='utf-8') as f:
            f.write(page_html)
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(page_html, 'html.parser')
            with open('booking_parsed.html', 'w', encoding='utf-8') as f:
                f.write(soup.prettify())
            # extract property-card divs and save
            try:
                extract_property_cards_from_soup(soup)
            except Exception:
                pass
        except Exception:
            pass
    except Exception as e:
        print('Selenium failed:', e)
    finally:
        try:
            if driver:
                driver.quit()
        except Exception:
            pass


def extract_property_cards_from_soup(soup, out_html='property_cards.html', out_json='property_cards.json'):
    """Extract all <div> elements whose class contains 'property-card' and save HTML + JSON.

    The function is defensive: Booking's markup varies, so it checks any class token
    containing 'property-card'.
    """
    cards = []
    # find all divs and filter by class tokens containing 'property-card'
    for div in soup.find_all('div'):
        classes = div.get('class') or []
        if any('property-card' in cls for cls in classes):
            cards.append(div)

    # save combined HTML
    with open(out_html, 'w', encoding='utf-8') as f:
        f.write('<!doctype html><html><head><meta charset="utf-8"></head><body>')
        for c in cards:
            f.write(str(c))
        f.write('</body></html>')

    # save JSON with minimal fields
    out = []
    for c in cards:
        text = c.get_text(separator=' ', strip=True)
        out.append({'html': str(c), 'text': text})
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ss', help='Search string (e.g. "Berlin, Germany")')
    parser.add_argument('--random', action='store_true', help='Pick a random location')
    parser.add_argument('--proxy', help='Proxy host:port or host:port:user:pass', default='31.59.20.176:6754')
    parser.add_argument('--no-headless', action='store_true', help='Run browser visible')
    args = parser.parse_args()

    samples = [
        'Paris, France', 'Cairo, Egypt', 'Tunis, Tunisia', 'New York, USA',
        'Hammamet, Tunisia', 'Tokyo, Japan', 'Berlin, Germany'
    ]

    if args.random:
        ss = random.choice(samples)
    elif args.ss:
        ss = args.ss
    else:
        ss = 'Hammamet, Tunisia'

    do_search_selenium(BASE_SEARCH_URL, ss, proxy=args.proxy, headless=not args.no_headless)


if __name__ == '__main__':
    main()

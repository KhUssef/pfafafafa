"""
flight_search.py
----------------
Exposes a single function: search_flights()

Usage:
    from flight_search import search_flights

    results = search_flights(
        from_city="Tunis",
        to_city="Paris",
        date="2025-09-15",
        serpapi_key="YOUR_KEY_HERE",
        # Optional:
        from_country="Tunisia",   # helps disambiguate city names
        to_country="France",
        adults=1,
        cabin="economy",          # economy | premium_economy | business | first
        max_stops=None,           # None=any, 0=direct, 1=max 1 stop
        currency="USD",
        return_date=None,         # "2025-09-22" for round-trip, None for one-way
    )

    for flight in results["flights"]:
        print(flight)

Airport data:
    Place airports.dat.txt (OurAirports/OpenFlights format) in the same
    directory as this script, or set AIRPORTS_DAT_PATH to its full path.
    Format per row:
        id, name, city, country, iata, icao, lat, lon, alt, tz, dst, tz_db, type, source

Get a free SerpApi key (100 searches/month, no credit card):
    https://serpapi.com/users/sign_up
"""

import csv
import os
import requests
from datetime import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Path to the airports data file — override via env var if needed
# ---------------------------------------------------------------------------
_DEFAULT_DAT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "airports.dat.txt")
AIRPORTS_DAT_PATH: str = os.getenv("AIRPORTS_DAT_PATH", _DEFAULT_DAT_PATH)


# ---------------------------------------------------------------------------
# Airport database loader
# Builds two indexes from the .dat.txt file:
#   _city_index    : "city_lower"            → first IATA found (fast default)
#   _city_country  : "city_lower|country_lower" → IATA (precise, disambiguation)
# Both are populated lazily on first call to _to_iata().
# ---------------------------------------------------------------------------
_city_index:   dict[str, str] = {}   # city → iata
_city_country: dict[str, str] = {}   # city|country → iata
_db_loaded = False


def _load_airports_db(path: str = AIRPORTS_DAT_PATH) -> None:
    """
    Parse the airports.dat.txt file and populate the lookup indexes.
    Columns (0-based):
        0  id
        1  name        e.g. "Goroka Airport"
        2  city        e.g. "Goroka"
        3  country     e.g. "Papua New Guinea"
        4  iata        e.g. "GKA"   (may be "" or "\\N" if unknown)
        5  icao
        6  lat  7 lon  8 alt  9 tz  10 dst  11 tz_db  12 type  13 source
    """
    global _db_loaded
    if _db_loaded:
        return

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Airport database not found at: {path}\n"
            "Download it from https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat\n"
            "and save as 'airports.dat.txt' next to flight_search.py, "
            "or set the AIRPORTS_DAT_PATH environment variable."
        )

    with open(path, encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if len(row) < 5:
                continue
            city    = row[2].strip().lower()
            country = row[3].strip().lower()
            iata    = row[4].strip().upper()
            name    = row[1].strip().lower()

            if not iata or iata == r"\N" or len(iata) != 3:
                continue  # skip entries with no valid IATA code

            # city → iata  (first occurrence wins; OurAirports orders by relevance)
            if city and city not in _city_index:
                _city_index[city] = iata

            # airport name → iata  (e.g. "heathrow" → "LHR")
            if name and name not in _city_index:
                _city_index[name] = iata

            # city|country → iata  (always store; last write wins for duplicates,
            # which is fine — same city/country should map to the same main airport)
            if city and country:
                _city_country[f"{city}|{country}"] = iata

    _db_loaded = True


def _to_iata(city: str, country: Optional[str] = None, serpapi_key: Optional[str] = None) -> str:
    """
    Convert a city (or airport name) to an IATA code.

    Resolution order:
        1. city + country key  → precise match (avoids London UK vs London CA)
        2. city-only key       → first airport in that city
        3. SerpApi airport search fallback (costs 1 API call)
        4. Raise ValueError
    """
    _load_airports_db()

    city_lc    = city.strip().lower()
    country_lc = country.strip().lower() if country else None

    # 1. Precise: city + country
    if country_lc:
        hit = _city_country.get(f"{city_lc}|{country_lc}")
        if hit:
            return hit

    # 2. City only
    hit = _city_index.get(city_lc)
    if hit:
        return hit

    # 3. SerpApi fallback
    if serpapi_key:
        query = f"{city} {country or ''}".strip()
        resp = requests.get(
            "https://serpapi.com/search.json",
            params={"engine": "google_flights_airports", "q": query, "api_key": serpapi_key},
            timeout=10,
        )
        if resp.ok:
            data = resp.json()
            airports = data.get("airports", [])
            if airports:
                iata = airports[0].get("iata") or airports[0].get("id")
                if iata:
                    return iata.upper()

    raise ValueError(
        f"Could not resolve '{city}' ({country or ''}) to an IATA code. "
        "Try passing the IATA code directly via from_iata / to_iata, "
        "or check the city spelling against the airports database."
    )


_CABIN_MAP = {
    "economy": "1",
    "premium_economy": "2",
    "premium economy": "2",
    "business": "3",
    "first": "4",
    "first class": "4",
}


def _fmt_duration(minutes: Optional[int]) -> str:
    if minutes is None:
        return "—"
    return f"{minutes // 60}h {minutes % 60:02d}m"


def _parse_offer(offer: dict, currency: str) -> dict:
    """Flatten a SerpApi flight offer into a clean dict."""
    legs = offer.get("flights", [])
    stops = max(len(legs) - 1, 0)
    first_leg = legs[0] if legs else {}
    last_leg = legs[-1] if legs else {}

    airlines = list(dict.fromkeys(f.get("airline", "") for f in legs if f.get("airline")))
    layovers = [
        {
            "airport": lv.get("name", ""),
            "duration": _fmt_duration(lv.get("duration")),
        }
        for lv in offer.get("layovers", [])
    ]

    segments = [
        {
            "from": f.get("departure_airport", {}).get("id", "?"),
            "from_name": f.get("departure_airport", {}).get("name", ""),
            "to": f.get("arrival_airport", {}).get("id", "?"),
            "to_name": f.get("arrival_airport", {}).get("name", ""),
            "departs": f.get("departure_airport", {}).get("time", ""),
            "arrives": f.get("arrival_airport", {}).get("time", ""),
            "duration": _fmt_duration(f.get("duration")),
            "airline": f.get("airline", ""),
            "flight_number": f.get("flight_number", ""),
            "aircraft": f.get("airplane", ""),
        }
        for f in legs
    ]

    return {
        "price": offer.get("price"),
        "currency": currency,
        "stops": stops,
        "stops_label": "Direct" if stops == 0 else f"{stops} stop{'s' if stops > 1 else ''}",
        "total_duration": _fmt_duration(offer.get("total_duration")),
        "airlines": airlines,
        "departs": first_leg.get("departure_airport", {}).get("time", ""),
        "arrives": last_leg.get("arrival_airport", {}).get("time", ""),
        "from_iata": first_leg.get("departure_airport", {}).get("id", ""),
        "to_iata": last_leg.get("arrival_airport", {}).get("id", ""),
        "layovers": layovers,
        "segments": segments,
        "booking_token": offer.get("booking_token", ""),
    }


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def search_flights(
    from_city: str,
    to_city: str,
    date: str,
    serpapi_key: str,
    *,
    from_country: Optional[str] = None,
    to_country: Optional[str] = None,
    from_iata: Optional[str] = None,   # override auto-lookup
    to_iata: Optional[str] = None,     # override auto-lookup
    adults: int = 1,
    cabin: str = "economy",
    max_stops: Optional[int] = None,   # None=any, 0=direct only, 1=max 1 stop
    currency: str = "USD",
    return_date: Optional[str] = None,
) -> dict:
    """
    Search for flights between two cities on a given date.

    Parameters
    ----------
    from_city     : Departure city name, e.g. "Tunis"
    to_city       : Arrival city name,   e.g. "Paris"
    date          : Departure date as "YYYY-MM-DD"
    serpapi_key   : Your SerpApi key (free at serpapi.com)
    from_country  : Country hint for disambiguation, e.g. "Tunisia"
    to_country    : Country hint for disambiguation, e.g. "France"
    from_iata     : Override with explicit IATA code, e.g. "TUN"
    to_iata       : Override with explicit IATA code, e.g. "CDG"
    adults        : Number of adult passengers (default 1)
    cabin         : "economy" | "premium_economy" | "business" | "first"
    max_stops     : None (any) | 0 (direct only) | 1 (max 1 stop)
    currency      : Currency code, e.g. "USD", "EUR", "TND"
    return_date   : Return date for round-trip, e.g. "2025-09-22" (None = one-way)

    Returns
    -------
    dict with keys:
        "from_iata"       : resolved departure IATA
        "to_iata"         : resolved arrival IATA
        "date"            : departure date
        "flights"         : list of flight offer dicts (sorted by price)
        "price_insights"  : dict with lowest_price, typical_range, price_level (may be None)
        "raw"             : full SerpApi response (for debugging)
    """
    # Validate date
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"date must be 'YYYY-MM-DD', got: {date!r}")

    # Resolve IATA codes
    dep = (from_iata or _to_iata(from_city, from_country, serpapi_key)).upper()
    arr = (to_iata or _to_iata(to_city, to_country, serpapi_key)).upper()

    # Build SerpApi params
    params: dict = {
        "engine": "google_flights",
        "api_key": serpapi_key,
        "departure_id": dep,
        "arrival_id": arr,
        "outbound_date": date,
        "type": "1" if return_date else "2",   # 1=round-trip, 2=one-way
        "adults": str(adults),
        "travel_class": _CABIN_MAP.get(cabin.lower(), "1"),
        "currency": currency,
        "hl": "en",
    }
    if return_date:
        params["return_date"] = return_date
    if max_stops is not None:
        params["stops"] = str(max_stops)

    # Call SerpApi
    resp = requests.get("https://serpapi.com/search.json", params=params, timeout=15)
    if not resp.ok:
        err = resp.json().get("error", f"HTTP {resp.status_code}")
        raise RuntimeError(f"SerpApi error: {err}")

    data = resp.json()

    # Parse results
    best = data.get("best_flights", [])
    other = data.get("other_flights", [])
    all_offers = [_parse_offer(o, currency) for o in best + other]
    all_offers.sort(key=lambda x: (x["price"] is None, x["price"] or 0))

    # Price insights
    pi = data.get("price_insights")
    insights = None
    if pi:
        insights = {
            "lowest_price": pi.get("lowest_price"),
            "typical_range": pi.get("typical_price_range"),
            "price_level": pi.get("price_level"),
            "currency": currency,
        }

    return {
        "from_iata": dep,
        "to_iata": arr,
        "date": date,
        "return_date": return_date,
        "flights": all_offers,
        "price_insights": insights,
        "raw": data,
    }


# ---------------------------------------------------------------------------
# Quick CLI test:  python flight_search.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os, json

    KEY = os.getenv("SERPAPI_KEY", "YOUR_KEY_HERE")

    results = search_flights(
        from_city="Tunis",
        to_city="Paris",
        date="2025-09-15",
        serpapi_key=KEY,
        from_country="Tunisia",
        to_country="France",
        currency="EUR",
    )

    print(f"\n✈  {results['from_iata']} → {results['to_iata']}  |  {results['date']}")
    print(f"   {len(results['flights'])} options found\n")

    for i, f in enumerate(results["flights"], 1):
        price = f"{f['price']} {f['currency']}" if f["price"] else "N/A"
        print(f"  {i}. {price}  |  {f['stops_label']}  |  {f['total_duration']}"
              f"  |  {f['departs']} → {f['arrives']}"
              f"  |  {', '.join(f['airlines'])}")

    if results["price_insights"]:
        pi = results["price_insights"]
        print(f"\n  💡 Lowest: {pi['lowest_price']} {pi['currency']}"
              f"  |  Level: {pi['price_level']}")

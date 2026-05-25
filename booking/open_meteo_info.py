from __future__ import annotations

import argparse
import json
from datetime import datetime
from typing import Any, Dict, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


WEATHER_CODE_LABELS = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def _http_get_json(base_url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    query = urlencode(params)
    url = f"{base_url}?{query}"
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=30) as response:
        data = response.read().decode("utf-8", errors="replace")
    return json.loads(data)


def geocode_city(city: str, country: Optional[str] = None) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "name": city,
        "count": 1,
        "language": "en",
        "format": "json",
    }
    if country:
        params["country"] = country

    payload = _http_get_json(GEOCODING_URL, params)
    results = payload.get("results", [])
    if not results:
        raise ValueError(f"No geocoding result found for city='{city}' country='{country or ''}'")
    return results[0]


def get_weather(lat: float, lon: float, days: int = 3) -> Dict[str, Any]:
    forecast_days = max(1, min(int(days), 16))
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
        "hourly": "temperature_2m,precipitation_probability,weather_code",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,sunrise,sunset",
        "timezone": "auto",
        "forecast_days": forecast_days,
    }
    return _http_get_json(FORECAST_URL, params)


def _label_weather_code(code: Any) -> str:
    try:
        return WEATHER_CODE_LABELS.get(int(code), "Unknown")
    except Exception:
        return "Unknown"


def print_human_summary(place: Dict[str, Any], weather: Dict[str, Any]) -> None:
    city = place.get("name", "")
    country = place.get("country", "")
    lat = place.get("latitude")
    lon = place.get("longitude")

    current = weather.get("current", {})
    daily = weather.get("daily", {})

    print(f"Location: {city}, {country} ({lat}, {lon})")
    print("Current:")
    print(
        f"  {current.get('time', '')} | "
        f"temp={current.get('temperature_2m')} C | "
        f"feels_like={current.get('apparent_temperature')} C | "
        f"humidity={current.get('relative_humidity_2m')}% | "
        f"wind={current.get('wind_speed_10m')} km/h | "
        f"precip={current.get('precipitation')} mm | "
        f"condition={_label_weather_code(current.get('weather_code'))}"
    )

    times = daily.get("time", []) or []
    max_t = daily.get("temperature_2m_max", []) or []
    min_t = daily.get("temperature_2m_min", []) or []
    precip = daily.get("precipitation_sum", []) or []
    codes = daily.get("weather_code", []) or []
    sunrise = daily.get("sunrise", []) or []
    sunset = daily.get("sunset", []) or []

    print("Daily forecast:")
    for i, day in enumerate(times):
        hi = max_t[i] if i < len(max_t) else "?"
        lo = min_t[i] if i < len(min_t) else "?"
        pr = precip[i] if i < len(precip) else "?"
        code = codes[i] if i < len(codes) else None
        sr = sunrise[i] if i < len(sunrise) else ""
        ss = sunset[i] if i < len(sunset) else ""
        print(
            f"  {day}: max={hi} C min={lo} C precip={pr} mm "
            f"condition={_label_weather_code(code)} sunrise={sr} sunset={ss}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch weather information from Open-Meteo.")
    parser.add_argument("--city", help="City name to geocode (for example: Dubai)")
    parser.add_argument("--country", help="Optional country filter for geocoding (ISO code or name)")
    parser.add_argument("--lat", type=float, help="Latitude (skip geocoding if provided with --lon)")
    parser.add_argument("--lon", type=float, help="Longitude (skip geocoding if provided with --lat)")
    parser.add_argument("--days", type=int, default=3, help="Forecast days (1-16), default=3")
    parser.add_argument("--json", action="store_true", help="Print raw JSON response")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.lat is not None and args.lon is not None:
        place = {
            "name": args.city or "Custom location",
            "country": args.country or "",
            "latitude": args.lat,
            "longitude": args.lon,
        }
    else:
        if not args.city:
            raise SystemExit("Provide --city (or both --lat and --lon).")
        place = geocode_city(args.city, args.country)

    weather = get_weather(float(place["latitude"]), float(place["longitude"]), args.days)

    if args.json:
        output = {
            "requested_at": datetime.utcnow().isoformat() + "Z",
            "location": {
                "name": place.get("name", ""),
                "country": place.get("country", ""),
                "latitude": place.get("latitude"),
                "longitude": place.get("longitude"),
            },
            "weather": weather,
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    print_human_summary(place, weather)


if __name__ == "__main__":
    main()

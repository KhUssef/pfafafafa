import csv
import requests
import time
import re
import os
import tempfile
import argparse
import html as html_lib
from tqdm import tqdm


WIKIVOYAGE_API = "https://en.wikivoyage.org/w/api.php"

INPUT_CSV = "new\\cities_missing_100k.csv"
OUTPUT_CSV = "new\\world-wikivoyage-descriptions.csv"

REQUEST_DELAY = 1.0
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; TravelBot/1.0; +https://example.com/bot)"
}


# ─────────────────────────────────────────────
# 🔎 Fetch full page content (sections)
# ─────────────────────────────────────────────

def fetch_wikivoyage_sections(title):
    params = {
        "action": "parse",
        "page": title,
        "prop": "sections|text|categories",
        "format": "json",
        "redirects": 1,
    }

    response = requests.get(
    WIKIVOYAGE_API,
    params=params,
    headers=HEADERS,
    timeout=20
    )    
    response.raise_for_status()
    return response.json()


def search_wikivoyage_titles(city, limit=5):
    params = {
        "action": "query",
        "list": "search",
        "srsearch": f'"{city}"',
        "srlimit": limit,
        "format": "json",
    }

    response = requests.get(
        WIKIVOYAGE_API,
        params=params,
        headers=HEADERS,
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()
    return [item.get("title", "").strip() for item in data.get("query", {}).get("search", [])]


# ─────────────────────────────────────────────
# 🧠 Extract relevant sections
# ─────────────────────────────────────────────

def extract_text_from_html(html):
    # Remove comments first to avoid carrying template/debug artifacts.
    html = re.sub(r"<!--.*?-->", " ", html, flags=re.DOTALL)

    # Remove style/script blocks so CSS/JS does not leak into text output.
    html = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)

    # Remove visual/media/layout-only blocks that should not pollute descriptions.
    for tag in ("figure", "table", "svg", "noscript"):
        html = re.sub(
            rf"<{tag}[^>]*>.*?</{tag}>",
            " ",
            html,
            flags=re.DOTALL | re.IGNORECASE,
        )

    # Remove standalone metadata/media tags.
    html = re.sub(r"<(img|meta|link)\b[^>]*>", " ", html, flags=re.IGNORECASE)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", html)

    # Decode entities like &gt;, &amp;, and numeric forms.
    text = html_lib.unescape(text)

    # Clean whitespace
    text = " ".join(text.split())

    return text


def extract_intro_and_understand(parsed):
    html = parsed.get("parse", {}).get("text", {}).get("*", "")
    sections = parsed.get("parse", {}).get("sections", [])

    # Extract intro (first paragraphs)
    intro_match = re.split(r"<h2", html, maxsplit=1)[0]
    intro_text = extract_text_from_html(intro_match)

    # Extract "Understand" section + subsections
    understand_text = ""

    for section in sections:
        if section.get("line", "").lower() == "understand":
            section_index = section.get("index")
            if section_index:
                # Find section HTML block via anchor
                pattern = rf"<h2.*?id=\"{section_index}\".*?</h2>(.*?)(<h2|$)"
                match = re.search(pattern, html, re.DOTALL)

                if match:
                    content_html = match.group(1)
                    understand_text = extract_text_from_html(content_html)

    return intro_text[:500], understand_text[:500]


def extract_geocrumbs_text(parsed):
    html = parsed.get("parse", {}).get("text", {}).get("*", "")
    match = re.search(
        r'<span[^>]*class="[^"]*ext-geocrumbs-breadcrumbs[^"]*"[^>]*>(.*?)</span>',
        html,
        re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return ""
    return extract_text_from_html(match.group(1))


def country_in_geocrumbs(parsed, country):
    geocrumbs = extract_geocrumbs_text(parsed)
    if not geocrumbs:
        return False

    normalized_country = " ".join((country or "").split()).casefold()
    normalized_geocrumbs = " ".join(geocrumbs.split()).casefold()
    return normalized_country in normalized_geocrumbs


def _normalize_for_contains(text):
    text = (text or "").replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().casefold()


def country_matches_page(parsed, country):
    normalized_country = _normalize_for_contains(country)
    if not normalized_country:
        return False

    geocrumbs = extract_geocrumbs_text(parsed)
    if geocrumbs:
        return normalized_country in _normalize_for_contains(geocrumbs)

    html = parsed.get("parse", {}).get("text", {}).get("*", "")
    intro_html = re.split(r"<h2", html, maxsplit=1)[0]
    intro_text = extract_text_from_html(intro_html)

    categories = parsed.get("parse", {}).get("categories", [])
    category_text = " ".join(
        (c.get("*") or "")
        for c in categories
        if isinstance(c, dict)
    )

    searchable = _normalize_for_contains(f"{intro_text} {category_text}")
    return normalized_country in searchable


# ─────────────────────────────────────────────
# 🔍 Query builder
# ─────────────────────────────────────────────

def build_queries(city, country):
    return [
        f"{city}, {country}",
        city,
        f"{city} ({country})",
    ]


def normalize_title_for_match(title):
    # Remove trailing parenthetical qualifier, e.g. "Safi (city)" -> "Safi"
    base = re.sub(r"\s*\([^)]*\)\s*$", "", title or "")
    return " ".join(base.split()).casefold()


def pick_best_title(titles, city):
    normalized_city = normalize_title_for_match(city)

    # Rule 1: if any candidate has "(city)", prefer it.
    for title in titles:
        if "(city)" in title.casefold():
            return title

    # Rule 2: otherwise choose exact title match from retrieved candidates.
    for title in titles:
        if normalize_title_for_match(title) == normalized_city:
            return title

    # Rule 3: fallback to first candidate (will be rejected by strict match check later).
    return titles[0] if titles else None


# ─────────────────────────────────────────────
# 🌐 Fetch description (intro + understand)
# ─────────────────────────────────────────────

def fetch_description(city, country):
    try:
        titles = search_wikivoyage_titles(city, limit=5)
        if not titles:
            return None

        ordered_titles = []
        preferred_title = pick_best_title(titles, city)
        if preferred_title:
            ordered_titles.append(preferred_title)
        for title in titles:
            if title != preferred_title:
                ordered_titles.append(title)

        for chosen_title in ordered_titles:
            if normalize_title_for_match(chosen_title) != normalize_title_for_match(city):
                continue

            parsed = fetch_wikivoyage_sections(chosen_title)
            if "error" in parsed:
                continue

            if not country_matches_page(parsed, country):
                continue

            intro, understand = extract_intro_and_understand(parsed)
            if not intro:
                continue

            combined = intro
            if understand:
                combined += " " + understand

            return clean_text(combined)

        return None
    except Exception as e:
        print(f"Error with {city}, {country}: {e}")
        return None


# ─────────────────────────────────────────────
# 🧹 Clean text
# ─────────────────────────────────────────────

def clean_text(text):
    text = " ".join(text.split())
    sentences = text.split(". ")
    cleaned = ". ".join(sentences[:3]).strip()
    if not cleaned:
        return ""
    if cleaned.endswith((".", "!", "?")):
        return cleaned
    return cleaned + "."


# ─────────────────────────────────────────────
# 📦 Main processing
# ─────────────────────────────────────────────

def create_unique_subcountries_temp_csv(input_csv):
    seen_subcountries = set()
    unique_rows = []

    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subcountry = (row.get("city") or "").strip()
            country = (row.get("country") or "").strip()

            if not subcountry or not country:
                continue

            key = subcountry.casefold()
            if key in seen_subcountries:
                continue

            seen_subcountries.add(key)
            unique_rows.append({
                "city": subcountry,
                "country": country,
            })

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        suffix=".csv",
        delete=False,
    ) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=["city", "country"])
        writer.writeheader()
        writer.writerows(unique_rows)
        temp_path = tmp.name

    return temp_path, len(unique_rows)

def process_dataset(input_csv, output_csv, test_mode=False):
    results = []
    temp_csv_path = None

    try:
        temp_csv_path, unique_count = create_unique_subcountries_temp_csv(input_csv)
        print(f"Using temporary unique-subcountries CSV ({unique_count} rows): {temp_csv_path}")

        with open(temp_csv_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))

            # TEST MODE → only 3 countries
            if test_mode:
                reader = reader[:3]

            for row in tqdm(reader, desc="Processing rows", unit="row"):
                city = strip_governorate((row.get("city") or "").strip())
                country = (row.get("country") or "").strip()

                if not city or not country:
                    continue

                description = fetch_description(city, country)

                results.append({
                    "city": city,
                    "country": country,
                    "description": description if description else "null"
                })

                time.sleep(REQUEST_DELAY)
    finally:
        if temp_csv_path and os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)

    # Save
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["city", "country", "description"]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Done → {len(results)} entries saved.")


def strip_governorate(text):
    cleaned = re.sub(r"\bGovernorate\b", " ", text or "", flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*,\s*", ", ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def redo_governorate_entries(csv_path=OUTPUT_CSV, output_csv=None):
    target_output = output_csv or csv_path

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fieldnames = rows[0].keys() if rows else ["city", "country", "description"]

    if not rows:
        print("No rows found in descriptions CSV.")
        return

    updated_count = 0

    for row in tqdm(rows, desc="Redoing governorate rows", unit="row"):
        subcountry = (row.get("city") or "").strip()
        country = (row.get("country") or "").strip()
        description = (row.get("description") or "").strip()

        if "governorate" not in f"{subcountry} {description}".casefold():
            continue

        cleaned_subcountry = strip_governorate(subcountry)
        if not cleaned_subcountry or not country:
            continue

        new_description = fetch_description(cleaned_subcountry, country)
        if new_description:
            row["description"] = new_description
            updated_count += 1

        time.sleep(REQUEST_DELAY)

    with open(target_output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Governorate redo done → {updated_count} rows updated.")


def run_single_example(subcountry, country):
    description = fetch_description(subcountry, country)
    print("\n🧪 Single example result:\n")
    print(f"subcountry: {subcountry}")
    print(f"country: {country}")
    print(f"description: {description if description else 'null'}")


# ─────────────────────────────────────────────
# 🧪 TEST FUNCTION
# ─────────────────────────────────────────────

def test_run():
    print("\n🧪 Running TEST MODE (3 entries)...\n")
    process_dataset(INPUT_CSV, OUTPUT_CSV, test_mode=True)


# ─────────────────────────────────────────────
# 🚀 ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Wikivoyage descriptions.")
    parser.add_argument("--single-subcountry", help="Run only one subcountry example")
    parser.add_argument("--single-country", help="Country for the single subcountry example")
    parser.add_argument("--test", action="store_true", help="Run in test mode (first 3 rows)")
    parser.add_argument(
        "--redo-governorates",
        action="store_true",
        help="Redo entries containing 'Governorate' from world-wikivoyage-descriptions.csv",
    )
    parser.add_argument(
        "--redo-output",
        help="Optional output CSV path for governorate redo (defaults to in-place)",
    )
    args = parser.parse_args()

    if args.redo_governorates:
        redo_governorate_entries(csv_path=OUTPUT_CSV, output_csv=args.redo_output)
    elif args.single_subcountry or args.single_country:
        if not args.single_subcountry or not args.single_country:
            parser.error("Both --single-subcountry and --single-country are required for single mode.")
        run_single_example(args.single_subcountry.strip(), args.single_country.strip())
    else:
        process_dataset(INPUT_CSV, OUTPUT_CSV, test_mode=args.test)
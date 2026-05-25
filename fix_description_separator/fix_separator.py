import csv
import os
from pathlib import Path
from tqdm import tqdm

# Define paths
SCRIPT_DIR = Path(__file__).parent
PARENT_DIR = SCRIPT_DIR.parent
WORLD_CITIES_CSV = PARENT_DIR / "world-cities.csv"
DESCRIPTIONS_CSV = PARENT_DIR / "world-wikivoyage-descriptions-redo.csv"

OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

BUGGY_ENTRIES_CSV = OUTPUT_DIR / "buggy_entries_fixed.csv"
COMPLETE_FIXED_CSV = OUTPUT_DIR / "world-wikivoyage-descriptions-fixed.csv"


def load_world_cities_mapping():
    """
    Create a mapping of subcountry -> country from world-cities.csv
    Also create a reverse mapping of country names for detecting merged entries
    """
    mapping = {}
    countries_set = set()
    try:
        with open(WORLD_CITIES_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                subcountry = (row.get("subcountry") or "").strip()
                country = (row.get("country") or "").strip()
                if subcountry and country:
                    # Store the most recent (or any valid) mapping for each subcountry
                    mapping[subcountry.casefold()] = country
                    countries_set.add(country)
    except FileNotFoundError:
        print(f"⚠️  Could not find {WORLD_CITIES_CSV}")
    
    return mapping, countries_set


def check_has_separator(subcountry_value, country_value, description_value=""):
    """
    Check if the entry has proper separation (country is not empty and is separate)
    Returns True if properly separated, False if it's a buggy entry
    
    Detects cases where:
    - Country field is empty or "null"
    - Country field looks like a description (too long, has many punctuation)
    - Description field is empty but country field has content
    """
    country = (country_value or "").strip()
    description = (description_value or "").strip()
    
    # If country field is empty or "null", it's buggy
    if not country or country.lower() == "null":
        return False
    
    # If country field looks like a description (very long, has multiple sentences)
    # Description fields typically have 50+ chars and descriptions.
    # Real country names are usually under 50 chars
    if len(country) > 100 and ("." in country or "," in country):
        # This looks like a description misplaced in the country field
        return False
    
    # If description is empty but country has a lot of text, it might be malformed
    if not description and len(country) > 200:
        return False
    
    return True


def fix_entry(subcountry_value, country_value, description_value, cities_mapping, countries_set):
    """
    Try to fix a buggy entry by:
    1. Looking it up directly in the cities mapping
    2. Extracting merged country from the subcountry field
    3. Recovering description from misplaced country field
    
    Returns (fixed_subcountry, fixed_country, fixed_description)
    """
    subcountry = (subcountry_value or "").strip()
    country = (country_value or "").strip()
    description = (description_value or "").strip()
    
    # If already properly separated, return as is
    if check_has_separator(subcountry, country, description):
        return subcountry, country, description
    
    # Check if the description is in the country field (malformed entry)
    recovered_description = description
    if not description and len(country) > 100 and ("." in country or "," in country):
        # The country field contains what should be the description
        recovered_description = country
        country = ""
    
    # Try direct lookup first
    lookup_key = subcountry.casefold()
    if lookup_key in cities_mapping:
        return subcountry, cities_mapping[lookup_key], recovered_description
    
    # Try to extract merged country from subcountry field
    # Check if subcountry ends with a known country name
    for country_name in countries_set:
        # Check if the subcountry ends with the country name (case-insensitive)
        if subcountry.lower().endswith(country_name.lower()):
            # Extract the city part
            city_part = subcountry[:-len(country_name)].strip()
            if city_part:
                return city_part, country_name, recovered_description
    
    # If still can't fix, return original
    return subcountry, country, recovered_description


def process_descriptions():
    """
    Process the descriptions CSV to find and fix buggy entries
    """
    print("📍 Loading world cities mapping...")
    cities_mapping, countries_set = load_world_cities_mapping()
    print(f"✅ Loaded {len(cities_mapping)} city->country mappings")
    print(f"✅ Found {len(countries_set)} unique countries")
    
    print("\n📖 Reading descriptions CSV...")
    all_rows = []
    buggy_rows = []
    
    try:
        with open(DESCRIPTIONS_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            for row in tqdm(reader, desc="Processing rows", unit="row"):
                subcountry = (row.get("subcountry") or "").strip()
                country = (row.get("country") or "").strip()
                description = (row.get("description") or "").strip()
                
                # Check if this is a buggy entry
                is_buggy = not check_has_separator(subcountry, country, description)
                
                if is_buggy:
                    # Try to fix it
                    fixed_subcountry, fixed_country, fixed_description = fix_entry(
                        subcountry, country, description, cities_mapping, countries_set
                    )
                    
                    fixed_row = {
                        "subcountry": fixed_subcountry,
                        "country": fixed_country,
                        "description": fixed_description if fixed_description else "null"
                    }
                    buggy_rows.append(fixed_row)
                    all_rows.append(fixed_row)
                else:
                    # Already fine, keep as is
                    fixed_row = {
                        "subcountry": subcountry,
                        "country": country,
                        "description": description if description else "null"
                    }
                    all_rows.append(fixed_row)
    
    except FileNotFoundError:
        print(f"❌ Could not find {DESCRIPTIONS_CSV}")
        return
    
    # Save buggy entries that were fixed
    print(f"\n💾 Saving {len(buggy_rows)} fixed buggy entries...")
    with open(BUGGY_ENTRIES_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subcountry", "country", "description"], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(buggy_rows)
    print(f"✅ Saved to {BUGGY_ENTRIES_CSV}")
    
    # Save complete fixed CSV
    print(f"\n💾 Saving complete fixed dataset ({len(all_rows)} entries)...")
    with open(COMPLETE_FIXED_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subcountry", "country", "description"], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"✅ Saved to {COMPLETE_FIXED_CSV}")
    
    print(f"\n📊 Summary:")
    print(f"   • Total entries: {len(all_rows)}")
    print(f"   • Buggy entries fixed: {len(buggy_rows)}")
    print(f"   • Properly formatted entries: {len(all_rows) - len(buggy_rows)}")


if __name__ == "__main__":
    process_descriptions()

import csv
import json
from pathlib import Path
from tqdm import tqdm

# Define paths
SCRIPT_DIR = Path(__file__).parent.parent
FIX_OUTPUT_DIR = SCRIPT_DIR / "fix_description_separator" / "output"
FIXED_DESC_CSV = FIX_OUTPUT_DIR / "world-wikivoyage-descriptions-fixed.csv"
ORIGINAL_DESC_CSV = SCRIPT_DIR / "world-wikivoyage-descriptions-redo.csv"

GROQ_DUAL_KEY_DIR = SCRIPT_DIR / "groq_dual_key"
GROQ_DUAL_KEY_CSV = GROQ_DUAL_KEY_DIR / "groq_tagged_cities.csv"
GROQ_ROOT_CSV = SCRIPT_DIR / "groq_tagged_cities.csv"

OUTPUT_DIR = SCRIPT_DIR / "fix_description_separator" / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

GROQ_MERGED_DUAL_KEY = OUTPUT_DIR / "groq_tagged_cities_with_descriptions.csv"
GROQ_MERGED_ROOT = OUTPUT_DIR / "groq_tagged_cities_root_with_descriptions.csv"
DESCRIPTION_CHANGES_LOG = OUTPUT_DIR / "description_changes.csv"


def load_descriptions_csv(csv_path):
    """
    Load descriptions CSV, handling the quoted format
    Returns dict: (subcountry, country) -> description
    """
    mapping = {}
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                subcountry = (row.get("subcountry") or "").strip()
                country = (row.get("country") or "").strip()
                description = (row.get("description") or "").strip()
                
                if subcountry and country:
                    key = (subcountry.lower(), country.lower())
                    mapping[key] = description
    except FileNotFoundError:
        print(f"⚠️  Could not find {csv_path}")
    
    return mapping


def load_groq_cities_csv(csv_path):
    """
    Load groq tagged cities CSV
    Returns list of (city, tags) tuples
    """
    rows = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                city = (row.get("city") or "").strip()
                tags = (row.get("tags") or "").strip()
                if city:
                    rows.append({"city": city, "tags": tags})
    except FileNotFoundError:
        print(f"⚠️  Could not find {csv_path}")
    
    return rows


def find_description_for_city(city, fixed_mapping, original_mapping):
    """
    Try to find a matching description for a city in the fixed mapping
    Returns the fixed description if found, otherwise the original, or "null"
    """
    # Try to find exact match (case-insensitive)
    for (subcountry, country), description in fixed_mapping.items():
        if subcountry.lower() == city.lower():
            return description
    
    # Try prefix match
    city_lower = city.lower()
    for (subcountry, country), description in fixed_mapping.items():
        if city_lower.startswith(subcountry.lower()):
            return description
    
    # Fallback to original mapping
    for (subcountry, country), description in original_mapping.items():
        if subcountry.lower() == city.lower():
            return description
    
    return "null"


def merge_groq_with_descriptions(groq_csv_path, fixed_mapping, original_mapping, output_path):
    """
    Merge groq tagged cities with fixed descriptions
    """
    groq_rows = load_groq_cities_csv(groq_csv_path)
    
    merged_rows = []
    for row in tqdm(groq_rows, desc=f"Processing {groq_csv_path.name}", unit="row"):
        city = row.get("city")
        tags = row.get("tags")
        
        # Find description
        description = find_description_for_city(city, fixed_mapping, original_mapping)
        
        merged_rows.append({
            "city": city,
            "tags": tags,
            "description": description
        })
    
    # Save merged CSV with proper quoting
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["city", "tags", "description"],
            quoting=csv.QUOTE_ALL
        )
        writer.writeheader()
        writer.writerows(merged_rows)
    
    print(f"✅ Saved merged CSV to {output_path}")
    return len(merged_rows)


def log_description_changes(fixed_mapping, original_mapping, output_path):
    """
    Log what descriptions changed between original and fixed
    """
    changes = []
    
    for (subcountry, country), fixed_desc in fixed_mapping.items():
        key = (subcountry.lower(), country.lower())
        
        # Find original
        original_desc = None
        for (orig_subctry, orig_country), orig_desc in original_mapping.items():
            if (orig_subctry.lower(), orig_country.lower()) == key:
                original_desc = orig_desc
                break
        
        # If descriptions differ, log it
        if original_desc and original_desc != fixed_desc:
            changes.append({
                "subcountry": subcountry,
                "country": country,
                "original_description": original_desc if original_desc else "null",
                "fixed_description": fixed_desc if fixed_desc else "null",
                "changed": "yes"
            })
    
    # Save changes log
    if changes:
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["subcountry", "country", "original_description", "fixed_description", "changed"],
                quoting=csv.QUOTE_ALL
            )
            writer.writeheader()
            writer.writerows(changes)
        print(f"✅ Saved {len(changes)} description changes to {output_path}")
    else:
        print("ℹ️  No description changes found")


def main():
    print("📍 Loading description files...\n")
    
    # Load mappings
    fixed_mapping = load_descriptions_csv(FIXED_DESC_CSV)
    original_mapping = load_descriptions_csv(ORIGINAL_DESC_CSV)
    
    print(f"✅ Loaded {len(fixed_mapping)} fixed descriptions")
    print(f"✅ Loaded {len(original_mapping)} original descriptions")
    
    # Log changes
    print("\n📊 Analyzing changes...")
    log_description_changes(fixed_mapping, original_mapping, DESCRIPTION_CHANGES_LOG)
    
    # Merge with groq dual key CSV
    if GROQ_DUAL_KEY_CSV.exists():
        print(f"\n🔀 Merging groq_dual_key/groq_tagged_cities.csv...")
        count = merge_groq_with_descriptions(
            GROQ_DUAL_KEY_CSV,
            fixed_mapping,
            original_mapping,
            GROQ_MERGED_DUAL_KEY
        )
        print(f"   Processed {count} cities")
    else:
        print(f"⚠️  {GROQ_DUAL_KEY_CSV} not found")
    
    # Merge with groq root CSV
    if GROQ_ROOT_CSV.exists():
        print(f"\n🔀 Merging root groq_tagged_cities.csv...")
        count = merge_groq_with_descriptions(
            GROQ_ROOT_CSV,
            fixed_mapping,
            original_mapping,
            GROQ_MERGED_ROOT
        )
        print(f"   Processed {count} cities")
    else:
        print(f"⚠️  {GROQ_ROOT_CSV} not found")
    
    print(f"\n📂 All output files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

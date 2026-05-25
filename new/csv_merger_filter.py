"""
Script to filter and merge CSVs based on city descriptions and matching criteria.

This script:
1. Creates CSV A: All cities from merged-world-wikivoyage-descriptions.csv with non-null descriptions
2. Creates CSV B1: Entries from new/groq_tagged_cities.csv that exist in CSV A
3. Creates CSV B2: Entries from groq_dual_key/groq_tagged_cities.csv that exist in CSV A
4. Creates CSV C: Entries from CSV A that do NOT exist in either B1 or B2
"""

import os
import pandas as pd

# Define paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Input files
CSV_A_SOURCE = os.path.join(SCRIPT_DIR, "merged-world-wikivoyage-descriptions.csv")
CSV_B1_SOURCE = os.path.join(SCRIPT_DIR, "groq_tagged_cities.csv")
CSV_B2_SOURCE = os.path.join(PROJECT_ROOT, "groq_dual_key", "groq_tagged_cities.csv")

# Output files
CSV_A_OUTPUT = os.path.join(SCRIPT_DIR, "filtered_cities_with_descriptions.csv")
CSV_B1_OUTPUT = os.path.join(SCRIPT_DIR, "groq_tagged_cities_in_descriptions.csv")
CSV_B2_OUTPUT = os.path.join(SCRIPT_DIR, "groq_dual_key_tagged_cities_in_descriptions.csv")
CSV_C_OUTPUT = os.path.join(SCRIPT_DIR, "cities_missing_from_groq_tags.csv")

def main():
    print("=" * 80)
    print("CSV Filtering and Merging Script")
    print("=" * 80)
    
    # Step 1: Create CSV A - cities with descriptions
    print("\n[Step 1] Loading merged-world-wikivoyage-descriptions.csv...")
    df_source = pd.read_csv(CSV_A_SOURCE)
    print(f"  Total rows: {len(df_source)}")
    
    # Filter for non-null descriptions
    df_a = df_source[df_source['description'].notna() & (df_source['description'].str.strip() != '')].copy()
    print(f"  Rows with descriptions: {len(df_a)}")
    
    df_a.to_csv(CSV_A_OUTPUT, index=False)
    print(f"  ✓ Saved to: {CSV_A_OUTPUT}")
    
    # Extract city names for matching
    cities_in_a = set(df_a['city'].unique())
    print(f"  Unique cities: {len(cities_in_a)}")
    
    # Step 2: Create CSV B1 - entries from new/groq_tagged_cities.csv that are in A
    print("\n[Step 2] Loading new/groq_tagged_cities.csv...")
    if os.path.exists(CSV_B1_SOURCE):
        df_b1_source = pd.read_csv(CSV_B1_SOURCE)
        print(f"  Total rows: {len(df_b1_source)}")
        
        df_b1 = df_b1_source[df_b1_source['city'].isin(cities_in_a)].copy()
        print(f"  Rows in filtered cities: {len(df_b1)}")
        
        df_b1.to_csv(CSV_B1_OUTPUT, index=False)
        print(f"  ✓ Saved to: {CSV_B1_OUTPUT}")
        
        cities_in_b1 = set(df_b1['city'].unique())
    else:
        print(f"  ✗ File not found: {CSV_B1_SOURCE}")
        cities_in_b1 = set()
    
    # Step 3: Create CSV B2 - entries from groq_dual_key/groq_tagged_cities.csv that are in A
    print("\n[Step 3] Loading groq_dual_key/groq_tagged_cities.csv...")
    if os.path.exists(CSV_B2_SOURCE):
        df_b2_source = pd.read_csv(CSV_B2_SOURCE)
        print(f"  Total rows: {len(df_b2_source)}")
        
        df_b2 = df_b2_source[df_b2_source['city'].isin(cities_in_a)].copy()
        print(f"  Rows in filtered cities: {len(df_b2)}")
        
        df_b2.to_csv(CSV_B2_OUTPUT, index=False)
        print(f"  ✓ Saved to: {CSV_B2_OUTPUT}")
        
        cities_in_b2 = set(df_b2['city'].unique())
    else:
        print(f"  ✗ File not found: {CSV_B2_SOURCE}")
        cities_in_b2 = set()
    
    # Step 4: Create CSV C - entries in A but NOT in B1 or B2
    print("\n[Step 4] Creating missing entries CSV...")
    cities_in_groq = cities_in_b1.union(cities_in_b2)
    cities_missing_from_groq = cities_in_a - cities_in_groq
    
    df_c = df_a[df_a['city'].isin(cities_missing_from_groq)].copy()
    print(f"  Cities in A but not in any groq CSV: {len(df_c)}")
    
    df_c.to_csv(CSV_C_OUTPUT, index=False)
    print(f"  ✓ Saved to: {CSV_C_OUTPUT}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"CSV A (cities with descriptions): {len(df_a)} rows")
    print(f"CSV B1 (groq_tagged_cities in A): {len(df_b1)} rows")
    print(f"CSV B2 (groq_dual_key_tagged_cities in A): {len(df_b2)} rows")
    print(f"CSV C (in A but not in B1/B2): {len(df_c)} rows")
    print(f"\nCities covered in groq CSVs: {len(cities_in_groq)} / {len(cities_in_a)}")
    print(f"Coverage: {len(cities_in_groq) / len(cities_in_a) * 100:.2f}%")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

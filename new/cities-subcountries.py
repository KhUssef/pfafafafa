import pandas as pd

# Load data
# ---------- Load data ----------
cities = pd.read_csv("new\simplemaps_worldcities_basicv1.901\worldcities.csv")  
subcountries = pd.read_csv("world-wikivoyage-descriptions-redo.csv")

# Normalize names
cities["name"] = cities["city"].str.strip().str.lower()
subcountries["name"] = subcountries["subcountry"].str.strip().str.lower()

subcountry_set = set(subcountries["name"])

# ---------- CONDITIONS ----------
cities["in_subcountry"] = cities["name"].isin(subcountry_set)
cities["pop_100k"] = cities["population"] >= 100000

# ---------- 1. KEEP FILE ----------
# condition: pop >= 100k OR appears in both
keep_df = cities[(cities["pop_100k"]) | (cities["in_subcountry"])]

# ---------- 2. MISSING FILE ----------
# condition: pop >= 100k AND NOT in subcountry
missing_df = cities[(cities["pop_100k"]) & (~cities["in_subcountry"])]

# ---------- CLEAN OUTPUT ----------
keep_df = keep_df[["city", "country", "population"]]
missing_df = missing_df[["city", "country", "population"]]

# ---------- SAVE ----------
keep_df.to_csv("cities_keep.csv", index=False)
missing_df.to_csv("cities_missing_100k.csv", index=False)

print("Saved:")
print(" - cities_keep.csv")
print(" - cities_missing_100k.csv")

print("\nCounts:")
print("keep:", len(keep_df))
print("missing:", len(missing_df))
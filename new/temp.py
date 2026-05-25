import pandas as pd

# Load data
# ---------- Load data ----------
cities = pd.read_csv("new\simplemaps_worldcities_basicv1.901\worldcities.csv")  
subcountries = pd.read_csv("world-wikivoyage-descriptions-redo.csv")

cities["name"] = cities["city_ascii"].str.strip().str.lower()
subcountries["name"] = subcountries["subcountry"].str.strip().str.lower()

# Sets
city_set = set(cities["name"])
subcountry_set = set(subcountries["name"])

# ---------- MATCH / MISMATCH ----------
cities_in_subcountries = cities[cities["name"].isin(subcountry_set)]
cities_not_in_subcountries = cities[~cities["name"].isin(subcountry_set)]

# ---------- POPULATION STATS ----------
def pop_stats(df):
    return {
        ">10000": (df["population"] > 10000).sum(),
        ">50000": (df["population"] > 50000).sum(),
        ">100000": (df["population"] > 100000).sum(),
        ">500000": (df["population"] > 500000).sum(),
    }

# ---------- RESULTS ----------
print("\n===== SUMMARY =====")
print("Cities in subcountries:", len(cities_in_subcountries))
print("Cities NOT in subcountries:", len(cities_not_in_subcountries))

print("\n-- Matched cities population breakdown --")
print(pop_stats(cities_in_subcountries))

print("\n-- Missing cities population breakdown --")
print(pop_stats(cities_not_in_subcountries))

# ---------- EXAMPLES ----------
print("\n===== EXAMPLES =====")

print("\nExample matched cities (in both datasets):")
print(
    cities_in_subcountries[["city", "population"]]
    .dropna()
    .sort_values("population", ascending=False)
    .head(10)
)

print("\nExample missing cities (NOT in subcountries):")
print(
    cities_not_in_subcountries[["city", "population"]]
    .dropna()
    .sort_values("population", ascending=False)
    .head(10)
)

print("\nExample subcountries not found in cities:")
subcountries_not_in_cities = subcountries[~subcountries["name"].isin(city_set)]
print(subcountries_not_in_cities["subcountry"].head(10).to_list())
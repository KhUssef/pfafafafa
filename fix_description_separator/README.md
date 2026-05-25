# Description Separator Fixer

## Purpose
This script identifies and fixes buggy entries in `world-wikivoyage-descriptions-redo.csv` where the country field is empty or missing, while using `world-cities.csv` to fill in the correct country mapping.

## What it does
1. Reads `world-cities.csv` to build a subcountry → country mapping
2. Scans `world-wikivoyage-descriptions-redo.csv` for entries with:
   - Empty country field, OR
   - Country field containing "null"
3. Fixes these entries using the mapping from world-cities.csv
4. Creates two output CSVs in the `output/` folder:
   - **buggy_entries_fixed.csv** - Only the entries that had the bug (now fixed)
   - **world-wikivoyage-descriptions-fixed.csv** - Complete dataset with all entries properly separated

## Output Files
- `output/buggy_entries_fixed.csv` - Fixed buggy entries only
- `output/world-wikivoyage-descriptions-fixed.csv` - Complete fixed dataset

## How to run
```bash
python fix_separator.py
```

## Requirements
- Python 3.6+
- tqdm (for progress bars)
- Files must exist in parent directory:
  - `world-cities.csv`
  - `world-wikivoyage-descriptions-redo.csv`

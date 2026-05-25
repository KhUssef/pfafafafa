import os
import sys
from pathlib import Path
import pandas as pd

# Ensure project root is importable
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from groq_dual_key import dual_key_runner as dkr

FIXED_CSV = PROJECT_ROOT / "fix_description_separator" / "output" / "world-wikivoyage-descriptions-fixed.csv"

# Output will be written to groq_dual_key/groq_tagged_cities.csv by the runner
def main():
    key1 = os.getenv("GROQ_API_KEY", "").strip()
    key2 = os.getenv("GROQ_API_KEY2", "").strip()

    if not key1 or not key2:
        print("⚠️  GROQ_API_KEY and GROQ_API_KEY2 are not set. The script will create the runner file but will not execute tagging.")
        print("If you want me to run tagging now, set the environment variables and re-run this script.")
        return

    if not FIXED_CSV.exists():
        print(f"⚠️  Fixed CSV not found at {FIXED_CSV}")
        return

    print(f"📍 Loading fixed CSV: {FIXED_CSV}")
    df = pd.read_csv(FIXED_CSV)
    print(f"Rows: {len(df)} Columns: {df.columns.tolist()}")

    dual_client = dkr.DualKeyGroqClient(key1, key2)
    results, stop_payload = dkr.process_full_dataset(df, dual_client)

    if stop_payload is None:
        print("✅ Tagging run completed. Output saved by the dual-key runner.")
    else:
        print("⚠️ Tagging run stopped due to daily limits. Partial output saved.")

    # After run, merge with descriptions using existing merge script
    merge_script = PROJECT_ROOT / "fix_description_separator" / "merge_groq_descriptions.py"
    if merge_script.exists():
        print("🔀 Now merging new groq output with descriptions...")
        os.system(f'python "{merge_script}"')
    else:
        print("⚠️ Merge script not found; please run merge_groq_descriptions.py manually.")

if __name__ == '__main__':
    main()

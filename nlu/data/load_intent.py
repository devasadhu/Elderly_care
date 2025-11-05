"""
Utility to merge multiple intent files for training
"""
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

INTENT_FILES = [
    "intents.json",
    "intents_augmented.json",
    "intents_emotional.json",
    "intents_followup.json"
]

def load_all_intents():
    all_intents = []
    for file_name in INTENT_FILES:
        file_path = DATA_DIR / file_name
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "intents" in data:
                    all_intents.extend(data["intents"])
    print(f"✅ Loaded {len(all_intents)} total intents from {len(INTENT_FILES)} files.")
    return {"intents": all_intents}

if __name__ == "__main__":
    merged = load_all_intents()
    with open(DATA_DIR / "intents_combined.json", "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print("💾 Saved merged dataset to intents_combined.json")

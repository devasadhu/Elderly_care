import os
import json
import glob
import random

# 💬 Simple Hinglish / Polite text augmenter
def augment_text(text):
    variants = [text]

    polite_prefixes = ["Please ", "Kindly ", "Could you please ", "Zara ", "Thoda "]
    if not text.lower().startswith(("please", "kindly", "could", "zara", "thoda")):
        variants.append(random.choice(polite_prefixes) + text)

    # Hinglish / Indian English tweaks
    variants.append(text.replace("hello", "namaste").replace("thank you", "dhanyavaad"))
    variants.append(text.capitalize())
    variants.append(text + " na?")
    return list(set(variants))

def augment_intents(data):
    """Apply augmentation to a single JSON file."""
    if isinstance(data, dict) and "intents" in data:
        intents_list = data["intents"]
        wrapped = True
    elif isinstance(data, list):
        intents_list = data
        wrapped = False
    else:
        raise ValueError("❌ Unsupported JSON format: expected list or { 'intents': [...] } structure")

    augmented_list = []
    for item in intents_list:
        augmented_item = item.copy()
        augmented_patterns = []
        for pattern in item.get("patterns", []) + item.get("examples", []):
            augmented_patterns.extend(augment_text(pattern))
        if augmented_patterns:
            augmented_item["patterns"] = list(set(augmented_patterns))
        augmented_list.append(augmented_item)

    return {"intents": augmented_list} if wrapped else augmented_list


if __name__ == "__main__":
    base_path = os.getcwd()
    intent_files = [
        f for f in glob.glob(os.path.join(base_path, "intents*.json"))
        if "augmented" not in f and not f.endswith("_auto.json")
    ]

    print(f"\n🧠 Found {len(intent_files)} intent files to augment:\n")
    for fpath in intent_files:
        print(f" → {os.path.basename(fpath)}")

    confirm = input("\nProceed to augment all files? (y/n): ").strip().lower()
    if confirm != "y":
        print("❌ Aborted.")
        exit()

    merged_data = []

    for fpath in intent_files:
        out_file = fpath.replace(".json", "_augmented_auto.json")
        print(f"\n⚙️ Augmenting: {os.path.basename(fpath)} → {os.path.basename(out_file)}")

        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        augmented_data = augment_intents(data)
        with open(out_file, "w", encoding="utf-8") as out:
            json.dump(augmented_data, out, ensure_ascii=False, indent=2)
        print(f"✅ Saved {os.path.basename(out_file)}")

        # For merging
        if isinstance(augmented_data, dict) and "intents" in augmented_data:
            merged_data.extend(augmented_data["intents"])
        elif isinstance(augmented_data, list):
            merged_data.extend(augmented_data)

    # Save merged file
    merged_path = os.path.join(base_path, "intents_all_augmented.json")
    with open(merged_path, "w", encoding="utf-8") as merged_out:
        json.dump({"intents": merged_data}, merged_out, ensure_ascii=False, indent=2)

    print(f"\n🗂️ Merged file saved as: {os.path.basename(merged_path)}")
    print("🎉 All files augmented and merged successfully!\n")

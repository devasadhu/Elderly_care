import os
import json
import random
from datetime import datetime

# Optional: use Ollama (local Mistral or other LLM)
try:
    import ollama
    USE_OLLAMA = True
except ImportError:
    USE_OLLAMA = False

# For synonym-based fallback augmentation
SYNONYM_MAP = {
    "hello": ["hi", "hey", "namaste", "good day"],
    "remind": ["remember", "note down", "set a reminder"],
    "medicine": ["tablet", "pills", "meds"],
    "sad": ["upset", "down", "low", "unhappy"],
    "call": ["ring", "phone", "talk to"],
    "thanks": ["thank you", "shukriya", "dhanyavaad", "much obliged"],
    "good": ["nice", "great", "fine", "wonderful"],
    "please": ["could you", "would you mind", "kindly"],
}

def augment_with_synonyms(sentence):
    """Replace words with synonyms randomly for simple data expansion."""
    words = sentence.split()
    new_words = []
    for w in words:
        lw = w.lower().strip(".,!?")
        if lw in SYNONYM_MAP and random.random() < 0.3:
            new_words.append(random.choice(SYNONYM_MAP[lw]))
        else:
            new_words.append(w)
    return " ".join(new_words)

def augment_with_ollama(prompt):
    """Use local LLM (e.g., Mistral) via Ollama to generate Hinglish / polite variants."""
    if not USE_OLLAMA:
        return None
    try:
        response = ollama.chat(model="mistral", messages=[
            {"role": "system", "content": "You are an AI that generates Hinglish or polite Indian-English versions of user queries."},
            {"role": "user", "content": prompt}
        ])
        text = response.get("message", {}).get("content", "")
        return text.strip()
    except Exception as e:
        print(f"[WARN] Ollama generation failed: {e}")
        return None

def augment_intents(input_path, output_path):
    """Main augmenter function."""
    with open(input_path, "r", encoding="utf-8") as f:
        intents = json.load(f)
        # Handle the case where the file has a top-level "intents" key
        if isinstance(intents, dict) and "intents" in intents:
            intents = intents["intents"]


    augmented_intents = []
    for item in intents:
        new_item = item if isinstance(item, dict) else {"text": item}
        new_patterns = set(item.get("examples", []))

        for ex in item.get("examples", []):
            # Hinglish/polite variants via Ollama
            if USE_OLLAMA:
                prompt = f"Generate 3 polite or Hinglish variations of this sentence for elderly users: '{ex}'"
                new_text = augment_with_ollama(prompt)
                if new_text:
                    for line in new_text.split("\n"):
                        if line.strip():
                            new_patterns.add(line.strip())
            else:
                # Simple synonym-based expansion
                new_patterns.add(augment_with_synonyms(ex))

        new_item["examples"] = list(new_patterns)
        augmented_intents.append(new_item)

    # Save new augmented file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(augmented_intents, f, indent=2, ensure_ascii=False)

    print(f"✅ Augmented data saved to {output_path}")
    print(f"🕒 Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Ollama: {USE_OLLAMA}")

if __name__ == "__main__":
    base_file = os.path.join(os.path.dirname(__file__), "../data/intents.json")
    output_file = os.path.join(os.path.dirname(__file__), "../data/intents_augmented_auto.json")
    augment_intents(base_file, output_file)

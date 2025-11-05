import torch
import torch.nn.functional as F
import json
import os
import re

from nlu.intent_classifier.model import IntentClassifier
from nlu.intent_classifier.dataset import IntentDataset
from nlu.config import MODEL_PATH, DATA_PATH


# =========================================================
# 🧠 Optional Local LLM Support (HuggingFace or Ollama)
# =========================================================
USE_LLM = False
LLM_SOURCE = None

# Try Transformers first
try:
    from transformers import pipeline
    LLM_SOURCE = "transformers"
    LLM_PIPE = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2")
    USE_LLM = True
    print("✅ HuggingFace LLM ready for fallback.")
except Exception:
    # Try Ollama next
    try:
        import subprocess

        def query_ollama(prompt, model="mistral"):
            try:
                result = subprocess.run(
                    ["ollama", "run", model, prompt],
                    capture_output=True, text=True, timeout=20
                )
                return result.stdout.strip()
            except Exception:
                return None

        test = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if test.returncode == 0:
            USE_LLM = True
            LLM_SOURCE = "ollama"
            print("✅ Ollama detected — using local LLM fallback.")
    except Exception:
        pass


# =========================================================
# 🧩 Intent Predictor Class (Hybrid Model + LLM)
# =========================================================
class IntentPredictor:
    """
    Hybrid Intent Predictor that combines:
      - Local trained intent classifier
      - Auto intent file selector (base/emotional/followup/smalltalk)
      - Optional LLM fallback when confidence is low
    """

    def __init__(self, model_path=None, vocab_size=5000, confidence_threshold=0.6):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.conf_threshold = confidence_threshold

        self.intent_files = {
            "base": os.path.join(DATA_PATH, "intents.json"),
            "emotional": os.path.join(DATA_PATH, "intents_emotional.json"),
            "followup": os.path.join(DATA_PATH, "intents_followup.json"),
            "smalltalk": os.path.join(DATA_PATH, "intents_smalltalk.json")
        }

        # Load dataset and tokenizer
        self.dataset = IntentDataset(intents_path=self.intent_files["base"])
        self.tokenizer = self.dataset.tokenizer

        # Load trained model
        if model_path is None:
            model_path = os.path.join(MODEL_PATH, "intent_classifier_best.pth")

        with open(self.intent_files["base"], "r", encoding="utf-8") as f:
            base_data = json.load(f)
        self.intent_labels = [item["intent"] for item in base_data]

        self.model = IntentClassifier(vocab_size=vocab_size, num_classes=len(self.intent_labels))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    # =========================================================
    # 🔍 Context-Aware Intent Loader
    # =========================================================
    def select_intent_file(self, text: str):
        text = text.lower()
        if any(w in text for w in ["sad", "lonely", "feel", "depressed", "upset"]):
            return self.intent_files["emotional"]
        elif any(w in text for w in ["remind", "remember", "call", "medicine", "appointment"]):
            return self.intent_files["followup"]
        elif any(w in text for w in ["hi", "hello", "thanks", "how are you", "bye", "good morning"]):
            return self.intent_files["smalltalk"]
        else:
            return self.intent_files["base"]

    # =========================================================
    # 🎯 Predict Intent
    # =========================================================
    def predict_intent(self, text, top_k=1):
        tokens = self.tokenizer(text)
        input_tensor = torch.tensor(tokens).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1)
            top_probs, top_idxs = probs.topk(top_k, dim=1)

        results = []
        for prob, idx in zip(top_probs[0], top_idxs[0]):
            results.append({"intent": self.intent_labels[idx.item()], "confidence": float(prob.item())})

        return results[0] if top_k == 1 else results

    # =========================================================
    # 💬 Get Response + LLM Fallback
    # =========================================================
    def get_response(self, text):
        selected_file = self.select_intent_file(text)
        with open(selected_file, "r", encoding="utf-8") as f:
            intents_data = json.load(f)

        prediction = self.predict_intent(text)
        intent_name = prediction["intent"]
        confidence = prediction["confidence"]

        matched_intent = next((i for i in intents_data if i["intent"] == intent_name), None)
        if matched_intent and "responses" in matched_intent:
            response_text = matched_intent["responses"][0]
        else:
            response_text = "I'm here for you. Could you tell me a bit more?"

        # 🔁 Low-confidence fallback to LLM
        if confidence < self.conf_threshold and USE_LLM:
            prompt = f"The user said: '{text}'. Respond kindly and naturally as an elderly care assistant."
            try:
                if LLM_SOURCE == "transformers":
                    response_text = LLM_PIPE(prompt, max_new_tokens=80)[0]["generated_text"]
                elif LLM_SOURCE == "ollama":
                    response_text = query_ollama(prompt)
            except Exception:
                response_text = "I'm listening carefully, please continue."

        return {
            "intent": intent_name,
            "confidence": confidence,
            "response": response_text,
            "used_file": os.path.basename(selected_file)
        }


# =========================================================
# 🧩 Simple Command-Line Demo
# =========================================================
if __name__ == "__main__":
    clf = IntentPredictor()
    print("🧠 ElderlyCare Hybrid NLU + LLM Ready!\n(Type 'exit' to quit)\n")

    while True:
        query = input("👵 You: ")
        if query.lower() in ["exit", "quit", "bye"]:
            print("🤖 Bot: Take care, have a peaceful day!")
            break
        output = clf.get_response(query)
        print(f"🎯 Intent: {output['intent']} ({output['confidence']:.2f}) | Source: {output['used_file']}")
        print(f"🤖 Bot: {output['response']}\n")

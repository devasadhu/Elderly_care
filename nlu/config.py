"""
Configuration file for Intent Classifier
"""

import os
from pathlib import Path

# ============================================================
# 📂 Base paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent  # points to nlu/
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ============================================================
# ⚙️ Model configuration
# ============================================================

MODEL_CONFIG = {
    "model_type": "bilstm",       # Options: "bilstm", "fast_cnn"
    "embedding_dim": 128,
    "hidden_dim": 256,
    "num_layers": 2,
    "dropout": 0.3,
    "num_filters": 128,
    "filter_sizes": [2, 3, 4],
    "max_vocab_size": 5000,
    "min_word_freq": 2,
    "max_seq_length": 30,
    "batch_size": 32,
    "num_epochs": 50,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "early_stopping_patience": 5,
    "optimizer": "adam",
    "scheduler": "reduce_on_plateau",
    "scheduler_patience": 3,
    "scheduler_factor": 0.5,
    "label_smoothing": 0.1,
    "gradient_clip": 5.0,
}

# Expose key parameters for easy import
EMBEDDING_DIM = MODEL_CONFIG["embedding_dim"]
HIDDEN_DIM = MODEL_CONFIG["hidden_dim"]
MAX_SEQ_LENGTH = MODEL_CONFIG["max_seq_length"]

# ============================================================
# 🧠 Intent labels (will be dynamically inferred later)
# ============================================================

INTENT_LABELS = [
    # Core
    "Emergency", "CallContact", "SetReminder", "GetWeather", "SetAlarm",
    "GetTime", "GetNews", "TellJoke", "PlayGame", "GeneralChat",
    
    # Emotional / Elderly-specific
    "ExpressEmotion", "ReassuranceNeeded", "Gratitude", "PraiseBot", "Goodbye",
    "Greeting", "Farewell", "Unknown", "HealthCheck", "EmergencyHelp",
    "CrisisSupport", "ReminderSetup", "CheckHealth", "MedicationInfo", 
    "CareTips", "LocationQuery", "SuicidalThought", "MoodCheck",

    # Follow-up intents
    "FollowUpReminder", "FollowUpHealth", "FollowUpContact", "FollowUpMood",
    "FollowUpActivity", "FollowUpMedication", "FollowUpEmergency", 
    "FollowUpCrisis", "FollowUpGeneral",

    # Small talk and empathy
    "SmallTalkJoke", "SmallTalkEncouragement", "SmallTalkMood",
    "SmallTalkBotIdentity", "SmallTalkWeather", "SmallTalkBotOpinion",
    "SmallTalkDailyRoutine",
]

NUM_INTENTS = len(INTENT_LABELS)
INTENT_TO_IDX = {intent: idx for idx, intent in enumerate(INTENT_LABELS)}
IDX_TO_INTENT = {idx: intent for idx, intent in enumerate(INTENT_LABELS)}

# ============================================================
# 🔄 Data merging / augmentation config
# ============================================================

AUGMENTATION_CONFIG = {
    "enabled": True,
    "synonym_replacement": 0.2,
    "random_insertion": 0.1,
    "random_swap": 0.1,
    "random_deletion": 0.1,
    "back_translation": False,
}

# This ensures all intent-related JSON files will be loaded and merged automatically.
MERGE_CONFIG = {
    "included_files": [
        "intents.json",
        "intents_augmented.json",
        "intents_smalltalk.json",
        "intents_emotional.json",
        "intents_augmented_auto.json",
        "intents_all_augmented.json",
        "intents_extended.json",
    ],
    "merged_output": DATA_DIR / "merged_intents.json",
}

# ============================================================
# 🗣️ Elderly speech normalization
# ============================================================

ELDERLY_SPEECH_PATTERNS = {
    "hesitations": ["um", "uh", "er", "hmm", "well"],
    "fillers": ["you know", "like", "I mean", "so"],
    "slow_speech": True,
    "repeat_words": True,
    "incomplete_sentences": True,
}

# ============================================================
# 🔤 Special tokens
# ============================================================

SPECIAL_TOKENS = {
    "PAD": "<PAD>",
    "UNK": "<UNK>",
    "BOS": "<BOS>",
    "EOS": "<EOS>",
}

# ============================================================
# 🧩 File paths and runtime configs
# ============================================================

PATHS = {
    # Keep backward compatibility
    "training_data": DATA_DIR / "intents.json",          # 👈 added back for older code references
    "training_data_dir": DATA_DIR,                       # new unified folder path
    "merged_data": DATA_DIR / "merged_intents.json",     # merged file
    "vocab_file": MODELS_DIR / "vocab.pkl",
    "model_checkpoint": MODELS_DIR / "intent_classifier_best.pth",
    "training_log": MODELS_DIR / "training_log.json",
}


LOGGING_CONFIG = {
    "log_every_n_steps": 10,
    "save_every_n_epochs": 5,
    "validate_every_n_epochs": 1,
}

INFERENCE_CONFIG = {
    "confidence_threshold": 0.7,
    "use_fallback_below": 0.5,
    "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
}

METRICS = ["accuracy", "precision", "recall", "f1_score", "confusion_matrix"]

# ============================================================
# 🧾 CLI summary
# ============================================================

if __name__ == "__main__":
    print("Intent Classifier Configuration")
    print("=" * 50)
    print(f"\nNumber of intents: {NUM_INTENTS}")
    print(f"Intent labels: {INTENT_LABELS}")
    print(f"\nModel type: {MODEL_CONFIG['model_type']}")
    print(f"Embedding dim: {EMBEDDING_DIM}")
    print(f"Hidden dim: {HIDDEN_DIM}")
    print(f"\nData directory: {DATA_DIR}")
    print(f"Models directory: {MODELS_DIR}")
    print("\nWill merge the following data files:")
    for f in MERGE_CONFIG["included_files"]:
        print(f"  • {f}")
    print("\nConfiguration loaded successfully!")

"""
PyTorch Dataset for Intent Classification
Loads and processes data from intents.json or merged datasets
"""

import json
import torch
from torch.utils.data import Dataset
import pickle
from collections import Counter
from pathlib import Path

from nlu.config import (
    INTENT_TO_IDX, 
    SPECIAL_TOKENS, 
    MODEL_CONFIG,
    ELDERLY_SPEECH_PATTERNS,
    PATHS
)


class IntentDataset(Dataset):
    """Dataset for intent classification"""
    
    def __init__(self, data_path, vocab=None, is_training=True):
        """
        Args:
            data_path: Path to intents.json or merged_intents.json
            vocab: Pre-built vocabulary (None for training)
            is_training: Whether this is training data
        """
        self.data_path = data_path
        self.is_training = is_training
        self.max_length = MODEL_CONFIG['max_seq_length']
        
        # Load data
        self.texts, self.labels = self._load_data()
        
        # Build or load vocabulary
        if vocab is None and is_training:
            self.vocab = self._build_vocab()
        else:
            self.vocab = vocab
        
        # Encode texts
        self.encoded_texts = [self._encode_text(text) for text in self.texts]
    
    def _load_data(self):
        """Load data from file (supports dict or list JSON structures)"""
        path = Path(self.data_path)
        if not path.exists():
            raise FileNotFoundError(f"❌ Data file not found: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        texts, labels = [], []

        # Handle both formats
        # 1️⃣ Format A: {"intents": [ {intent, examples}, ... ]}
        # 2️⃣ Format B: [ {intent, examples}, ... ]
        # 3️⃣ Format C: [ {"text": "...", "intent": "..."} ] (flattened)
        intents_list = []
        if isinstance(data, dict) and "intents" in data:
            intents_list = data["intents"]
        elif isinstance(data, list):
            intents_list = data
        else:
            raise ValueError(f"Unsupported JSON structure in {path}")

        for intent_data in intents_list:
            # Case 1: has nested examples (standard structure)
            if isinstance(intent_data, dict) and "intent" in intent_data and "examples" in intent_data:
                intent = intent_data["intent"]
                if intent not in INTENT_TO_IDX:
                    print(f"Warning: Intent '{intent}' not in INTENT_TO_IDX, skipping...")
                    continue

                label_idx = INTENT_TO_IDX[intent]
                for example in intent_data["examples"]:
                    texts.append(example.strip())
                    labels.append(label_idx)

            # Case 2: flattened format
            elif isinstance(intent_data, dict) and "text" in intent_data and "intent" in intent_data:
                intent = intent_data["intent"]
                if intent not in INTENT_TO_IDX:
                    print(f"Warning: Intent '{intent}' not in INTENT_TO_IDX, skipping...")
                    continue

                label_idx = INTENT_TO_IDX[intent]
                texts.append(intent_data["text"].strip())
                labels.append(label_idx)

        print(f"Loaded {len(texts)} examples from {path}")
        return texts, labels
    
    def _build_vocab(self):
        """Build vocabulary from training texts"""
        word_freq = Counter()
        
        for text in self.texts:
            tokens = self._tokenize(text)
            word_freq.update(tokens)
        
        # Create vocabulary with special tokens
        vocab = {
            SPECIAL_TOKENS['PAD']: 0,
            SPECIAL_TOKENS['UNK']: 1,
        }
        
        # Add words above minimum frequency
        min_freq = MODEL_CONFIG['min_word_freq']
        max_vocab = MODEL_CONFIG['max_vocab_size']
        
        idx = len(vocab)
        for word, freq in word_freq.most_common():
            if freq >= min_freq and idx < max_vocab:
                vocab[word] = idx
                idx += 1
        
        print(f"Built vocabulary: {len(vocab)} words")
        print(f"Most common words: {word_freq.most_common(10)}")
        
        return vocab
    
    def _tokenize(self, text):
        """Tokenize text (simple whitespace tokenization)"""
        text = text.lower().strip()
        tokens = text.split()
        
        # Handle elderly speech patterns
        if ELDERLY_SPEECH_PATTERNS['hesitations']:
            hesitations = set(ELDERLY_SPEECH_PATTERNS['hesitations'])
            tokens = [t for t in tokens if t not in hesitations]
        
        return tokens
    
    def _encode_text(self, text):
        """Convert text to indices"""
        tokens = self._tokenize(text)
        unk_idx = self.vocab[SPECIAL_TOKENS['UNK']]
        
        indices = [self.vocab.get(token, unk_idx) for token in tokens]
        
        # Pad or truncate
        if len(indices) < self.max_length:
            pad_idx = self.vocab[SPECIAL_TOKENS['PAD']]
            indices += [pad_idx] * (self.max_length - len(indices))
        else:
            indices = indices[:self.max_length]
        
        return indices
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        """Get single item"""
        return {
            'input_ids': torch.tensor(self.encoded_texts[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
            'text': self.texts[idx]
        }
    
    def save_vocab(self, path=None):
        """Save vocabulary to file"""
        if path is None:
            path = PATHS['vocab_file']
        
        with open(path, 'wb') as f:
            pickle.dump(self.vocab, f)
        
        print(f"Vocabulary saved to {path}")
    
    @staticmethod
    def load_vocab(path=None):
        """Load vocabulary from file"""
        if path is None:
            path = PATHS['vocab_file']
        
        with open(path, 'rb') as f:
            vocab = pickle.load(f)
        
        print(f"Vocabulary loaded: {len(vocab)} words")
        return vocab
    
    def get_class_weights(self):
        """Calculate class weights for imbalanced data"""
        label_counts = Counter(self.labels)
        total = len(self.labels)
        
        weights = []
        for i in range(len(INTENT_TO_IDX)):
            count = label_counts.get(i, 1)
            weight = total / (len(INTENT_TO_IDX) * count)
            weights.append(weight)
        
        return torch.FloatTensor(weights)


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'labels': labels
    }


if __name__ == "__main__":
    # Test the dataset
    print("Testing IntentDataset...")
    
    dataset = IntentDataset(PATHS['training_data'])
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Vocabulary size: {len(dataset.vocab)}")
    
    # Test a sample
    sample = dataset[0]
    print(f"\nSample:")
    print(f"Text: {sample['text']}")
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Label: {sample['labels'].item()}")
    
    # Test DataLoader
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset, 
        batch_size=4, 
        shuffle=True,
        collate_fn=collate_fn
    )
    
    batch = next(iter(dataloader))
    print(f"\nBatch:")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    
    print("\n✅ Dataset test successful!")

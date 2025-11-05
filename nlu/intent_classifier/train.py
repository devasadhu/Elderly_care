"""
Training script for Intent Classification Model
Trains the BiLSTM model on elderly care intents
"""
# ✅ FIXED IMPORTS (absolute, not relative)
from nlu.intent_classifier.model import IntentClassifier, FastIntentClassifier
from nlu.intent_classifier.dataset import IntentDataset, collate_fn

from nlu.config import (
    MODEL_CONFIG,
    PATHS,
    INTENT_LABELS,
    IDX_TO_INTENT,
    LOGGING_CONFIG,
    INFERENCE_CONFIG
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pickle
import json
import glob
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from datetime import datetime


class IntentTrainer:
    """Train intent classification model"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # ✅ Load and merge all JSON data files
        print("\nLoading training data...")
        merged_path = self._merge_intent_jsons()

        # Create dataset from merged file
        self.full_dataset = IntentDataset(merged_path, is_training=True)

        # Split into train/val
        train_size = int(0.8 * len(self.full_dataset))
        val_size = len(self.full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            self.full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        # Share vocabulary with validation set
        self.val_dataset.dataset.vocab = self.full_dataset.vocab

        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")

        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=MODEL_CONFIG['batch_size'],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=2
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=MODEL_CONFIG['batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2
        )

        # Initialize model
        vocab_size = len(self.full_dataset.vocab)
        num_intents = len(INTENT_LABELS)

        if MODEL_CONFIG['model_type'] == 'bilstm':
            self.model = IntentClassifier(
                vocab_size=vocab_size,
                embedding_dim=MODEL_CONFIG['embedding_dim'],
                hidden_dim=MODEL_CONFIG['hidden_dim'],
                num_intents=num_intents,
                dropout=MODEL_CONFIG['dropout'],
                num_layers=MODEL_CONFIG['num_layers']
            )
        else:
            self.model = FastIntentClassifier(
                vocab_size=vocab_size,
                embedding_dim=MODEL_CONFIG['embedding_dim'],
                num_filters=MODEL_CONFIG['num_filters'],
                filter_sizes=MODEL_CONFIG['filter_sizes'],
                num_intents=num_intents,
                dropout=MODEL_CONFIG['dropout']
            )

        self.model = self.model.to(self.device)

        print(f"\nModel: {MODEL_CONFIG['model_type']}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Loss function (with class weights for imbalanced data)
        class_weights = self.full_dataset.get_class_weights().to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=MODEL_CONFIG['learning_rate'],
            weight_decay=MODEL_CONFIG['weight_decay']
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=MODEL_CONFIG['scheduler_patience'],
            factor=MODEL_CONFIG['scheduler_factor'],
            verbose=True
        )

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }

        self.best_val_acc = 0.0
        self.patience_counter = 0

    # ============================================================
    # ✅ NEW: Merge all JSON intent data automatically
    # ============================================================

    def _merge_intent_jsons(self):
        data_dir = Path(PATHS["training_data"]).parent
        all_data = []
        print(f"Scanning directory: {data_dir}")
        
        json_files = sorted([p for p in data_dir.glob("*.json")])

        if not json_files:
            print("No .json files found in data directory.")
            merged_path = data_dir / "merged_intents.json"
            with open(merged_path, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2, ensure_ascii=False)
                print(f"Saved empty merged file to: {merged_path}")
                return merged_path

        processed_files = 0
        for file_path in json_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Handle both list-based and dict-based JSONs
                if isinstance(data, dict) and "intents" in data and isinstance(data["intents"], list):
                    flat = []
                    for intent_block in data["intents"]:
                        intent_name = intent_block.get("intent")
                        for example in intent_block.get("examples", []):
                            flat.append({"intent": intent_name, "text": example})
                    data = flat

                if isinstance(data, list):
                    valid_items = []
                    for item in data:
                        if isinstance(item, dict) and "intent" in item:
                            if "text" in item:
                                valid_items.append({"intent": item["intent"], "text": item["text"]})
                            elif "examples" in item and isinstance(item["examples"], list):
                                for ex in item["examples"]:
                                    valid_items.append({"intent": item["intent"], "text": ex})
                            else:
                                text_val = item.get("text") or item.get("example") or item.get("utterance")
                                if text_val:
                                    valid_items.append({"intent": item["intent"], "text": text_val})
                    if valid_items:
                        all_data.extend(valid_items)
                        processed_files += 1
                    else:
                        print(f"Skipping {file_path.name}: list found but no valid intent records")
                else:
                    print(f"Skipping {file_path.name}: unsupported JSON structure")

            except json.JSONDecodeError:
                print(f"Error decoding {file_path.name}, skipping.")
            except Exception as e:
                print(f"Unexpected error while processing {file_path.name}: {e}")

        merged_path = data_dir / "merged_intents.json"
        with open(merged_path, "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)
            
            files_count = len(json_files)
            print(f"Loaded {len(all_data)} samples from {files_count} JSON files (processed {processed_files} files).")
            print(f"Saved merged data to: {merged_path}")
            return merged_path

    # ============================================================
    # Existing training logic below (unchanged)
    # ============================================================

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Training")

        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.optimizer.zero_grad()

            if MODEL_CONFIG['model_type'] == 'bilstm':
                outputs, _ = self.model(input_ids)
            else:
                outputs = self.model(input_ids)

            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), MODEL_CONFIG['gradient_clip'])
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{100.*correct/total:.2f}%"})

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy

    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                if MODEL_CONFIG['model_type'] == 'bilstm':
                    outputs, _ = self.model(input_ids)
                else:
                    outputs = self.model(input_ids)

                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy

    def train(self):
        """Full training loop"""
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)

        num_epochs = MODEL_CONFIG['num_epochs']

        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 60)

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc)
            self.history['learning_rate'].append(current_lr)

            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.6f}")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                self.save_checkpoint(is_best=True)
                print(f"✓ New best model! Validation Accuracy: {val_acc:.2f}%")
            else:
                self.patience_counter += 1

            if self.patience_counter >= MODEL_CONFIG['early_stopping_patience']:
                print(f"\n⚠ Early stopping triggered after {epoch} epochs")
                break

        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Best Validation Accuracy: {self.best_val_acc:.2f}%")
        print("=" * 60)

        self.save_history()
        self.save_artifacts()

    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'vocab_size': len(self.full_dataset.vocab),
            'num_intents': len(INTENT_LABELS),
            'model_config': MODEL_CONFIG,
            'best_val_acc': self.best_val_acc
        }

        if is_best:
            path = PATHS['model_checkpoint']
            torch.save(checkpoint, path)
            print(f"Model saved to {path}")

    def save_history(self):
        """Save training history"""
        history_path = PATHS['training_log']
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to {history_path}")

    def save_artifacts(self):
        """Save vocabulary and label decoder"""
        self.full_dataset.save_vocab()
        label_decoder_path = Path(PATHS['vocab_file']).parent / 'label_decoder.pkl'
        with open(label_decoder_path, 'wb') as f:
            pickle.dump(IDX_TO_INTENT, f)
        print(f"Label decoder saved to {label_decoder_path}")


def main():
    """Main training function"""
    print("=" * 60)
    print("Intent Classifier Training")
    print("=" * 60)

    trainer = IntentTrainer()
    trainer.train()
    print("\n✅ Training complete! Now you can use inference.py to test.")


if __name__ == "__main__":
    main()

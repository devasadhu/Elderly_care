"""
Evaluation script for Intent Classification Model
Comprehensive metrics and confusion matrix visualization
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import pickle
from pathlib import Path

from model import IntentClassifier, FastIntentClassifier
from dataset import IntentDataset, collate_fn
from torch.utils.data import DataLoader
from nlu.config import (
MODEL_CONFIG, PATHS, INTENT_LABELS, IDX_TO_INTENT
)

class ModelEvaluator:
    """Evaluate trained model performance"""
    
    def __init__(self, model_path=None, test_data_path=None):
        """Initialize evaluator"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Paths
        if model_path is None:
            model_path = PATHS['model_checkpoint']
        if test_data_path is None:
            test_data_path = PATHS['training_data']  # Use full dataset for now
        
        # Load vocabulary
        vocab_path = PATHS['vocab_file']
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
        
        # Load label decoder
        label_decoder_path = vocab_path.parent / 'label_decoder.pkl'
        with open(label_decoder_path, 'rb') as f:
            self.label_decoder = pickle.load(f)
        
        # Load test dataset
        self.test_dataset = IntentDataset(
            test_data_path, 
            vocab=self.vocab, 
            is_training=False
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=MODEL_CONFIG['batch_size'],
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        
        vocab_size = len(self.vocab)
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
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Test dataset: {len(self.test_dataset)} samples")
    
    def predict(self):
        """Get predictions for test dataset"""
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels']
                
                if MODEL_CONFIG['model_type'] == 'bilstm':
                    outputs, _ = self.model(input_ids)
                else:
                    outputs = self.model(input_ids)
                
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)
    
    def calculate_metrics(self, predictions, labels):
        """Calculate evaluation metrics"""
        # Overall accuracy
        accuracy = accuracy_score(labels, predictions)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None, labels=range(len(INTENT_LABELS))
        )
        
        # Macro averages
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1
        }
    
    def print_metrics(self, metrics):
        """Print metrics in a nice format"""
        print("\n" + "="*70)
        print("EVALUATION METRICS")
        print("="*70)
        
        print(f"\n📊 Overall Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"   Macro Precision: {metrics['macro_precision']*100:.2f}%")
        print(f"   Macro Recall: {metrics['macro_recall']*100:.2f}%")
        print(f"   Macro F1-Score: {metrics['macro_f1']*100:.2f}%")
        
        print("\n" + "-"*70)
        print(f"{'Intent':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}")
        print("-"*70)
        
        for i, intent in enumerate(INTENT_LABELS):
            print(f"{intent:<20} "
                  f"{metrics['precision'][i]*100:>8.2f}%   "
                  f"{metrics['recall'][i]*100:>8.2f}%   "
                  f"{metrics['f1'][i]*100:>8.2f}%   "
                  f"{metrics['support'][i]:>5d}")
        
        print("="*70)
    
    def plot_confusion_matrix(self, predictions, labels, save_path=None):
        """Plot confusion matrix"""
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=INTENT_LABELS,
            yticklabels=INTENT_LABELS,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Intent Classification Confusion Matrix', fontsize=16, pad=20)
        plt.ylabel('True Intent', fontsize=12)
        plt.xlabel('Predicted Intent', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path is None:
            save_path = PATHS['model_checkpoint'].parent / 'confusion_matrix.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Confusion matrix saved to {save_path}")
        plt.close()
    
    def plot_per_class_performance(self, metrics, save_path=None):
        """Plot per-class performance metrics"""
        x = np.arange(len(INTENT_LABELS))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.bar(x - width, metrics['precision']*100, width, label='Precision', alpha=0.8)
        ax.bar(x, metrics['recall']*100, width, label='Recall', alpha=0.8)
        ax.bar(x + width, metrics['f1']*100, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Intent', fontsize=12)
        ax.set_ylabel('Score (%)', fontsize=12)
        ax.set_title('Per-Intent Performance Metrics', fontsize=16, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(INTENT_LABELS, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 105])
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = PATHS['model_checkpoint'].parent / 'per_class_performance.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Per-class performance plot saved to {save_path}")
        plt.close()
    
    def analyze_errors(self, predictions, labels, probabilities, top_n=10):
        """Analyze most common errors"""
        print("\n" + "="*70)
        print("ERROR ANALYSIS - Most Common Misclassifications")
        print("="*70)
        
        # Find all errors
        errors = []
        for i, (pred, true) in enumerate(zip(predictions, labels)):
            if pred != true:
                text = self.test_dataset.texts[i]
                confidence = probabilities[i][pred]
                errors.append({
                    'text': text,
                    'true': IDX_TO_INTENT[true],
                    'predicted': IDX_TO_INTENT[pred],
                    'confidence': confidence
                })
        
        if len(errors) == 0:
            print("\n✨ Perfect! No errors found.")
            return
        
        print(f"\nTotal Errors: {len(errors)} / {len(predictions)} "
              f"({len(errors)/len(predictions)*100:.2f}%)")
        
        print(f"\nTop {min(top_n, len(errors))} Most Confident Errors:")
        print("-"*70)
        
        # Sort by confidence (most confident wrong predictions)
        errors_sorted = sorted(errors, key=lambda x: x['confidence'], reverse=True)
        
        for i, error in enumerate(errors_sorted[:top_n], 1):
            print(f"\n{i}. Text: \"{error['text']}\"")
            print(f"   True Intent: {error['true']}")
            print(f"   Predicted: {error['predicted']} (confidence: {error['confidence']*100:.2f}%)")
    
    def evaluate(self, plot=True):
        """Run full evaluation"""
        print("\n🔍 Running Model Evaluation...")
        
        # Get predictions
        predictions, labels, probabilities = self.predict()
        
        # Calculate metrics
        metrics = self.calculate_metrics(predictions, labels)
        
        # Print metrics
        self.print_metrics(metrics)
        
        # Plot visualizations
        if plot:
            self.plot_confusion_matrix(predictions, labels)
            self.plot_per_class_performance(metrics)
        
        # Error analysis
        self.analyze_errors(predictions, labels, probabilities)
        
        return metrics


def main():
    """Main evaluation function"""
    print("="*70)
    print("HeyCare Intent Classifier - Model Evaluation")
    print("="*70)
    
    # Create evaluator
    evaluator = ModelEvaluator()
    
    # Run evaluation
    metrics = evaluator.evaluate(plot=True)
    
    # Check if target accuracy is met
    target_acc = 0.95
    if metrics['accuracy'] >= target_acc:
        print(f"\n✅ Target accuracy of {target_acc*100}% achieved!")
    else:
        print(f"\n⚠️  Target accuracy of {target_acc*100}% not yet reached.")
        print(f"   Current: {metrics['accuracy']*100:.2f}%")
        print(f"   Gap: {(target_acc - metrics['accuracy'])*100:.2f}%")
        print("\n💡 Suggestions:")
        print("   - Add more training data")
        print("   - Try data augmentation")
        print("   - Adjust model hyperparameters")


if __name__ == "__main__":
    main()
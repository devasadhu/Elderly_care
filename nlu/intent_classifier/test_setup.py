"""
Test script to verify NLU module setup
Run this before training to ensure everything is configured correctly
"""
import sys
from pathlib import Path

# Ensure the current directory (intent_classifier) and its parent (nlu) are in the import path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(parent_dir))


def test_imports():
    """Test if all required libraries are installed"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not installed")
        return False
    
    try:
        import numpy
        print(f"✓ NumPy {numpy.__version__}")
    except ImportError:
        print("✗ NumPy not installed")
        return False
    
    try:
        import sklearn
        print(f"✓ Scikit-learn {sklearn.__version__}")
    except ImportError:
        print("✗ Scikit-learn not installed")
        return False
    
    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
    except ImportError:
        print("✗ Matplotlib not installed")
        return False
    
    try:
        import seaborn
        print(f"✓ Seaborn {seaborn.__version__}")
    except ImportError:
        print("✗ Seaborn not installed")
        return False
    
    try:
        import tqdm
        print(f"✓ tqdm {tqdm.__version__}")
    except ImportError:
        print("✗ tqdm not installed")
        return False
    
    return True


def test_config():
    """Test if config.py loads correctly"""
    print("\nTesting configuration...")
    
    try:
        from config import (
            MODEL_CONFIG, INTENT_LABELS, PATHS, 
            NUM_INTENTS, INTENT_TO_IDX
        )
        print(f"✓ Config loaded successfully")
        print(f"  - Number of intents: {NUM_INTENTS}")
        print(f"  - Model type: {MODEL_CONFIG['model_type']}")
        print(f"  - Batch size: {MODEL_CONFIG['batch_size']}")
        return True
    except Exception as e:
        print(f"✗ Config error: {e}")
        return False


def test_data_file():
    """Test if training data exists"""
    print("\nTesting data files...")
    
    try:
        from config import PATHS
        
        if PATHS['training_data'].exists():
            print(f"✓ Training data found: {PATHS['training_data']}")
            
            import json
            with open(PATHS['training_data'], 'r') as f:
                data = json.load(f)
            
            num_intents = len(data['intents'])
            total_examples = sum(len(intent['examples']) for intent in data['intents'])
            
            print(f"  - Intents: {num_intents}")
            print(f"  - Total examples: {total_examples}")
            print(f"  - Examples per intent: {total_examples // num_intents}")
            
            return True
        else:
            print(f"✗ Training data not found at {PATHS['training_data']}")
            return False
            
    except Exception as e:
        print(f"✗ Data file error: {e}")
        return False


def test_model():
    """Test if model can be instantiated"""
    print("\nTesting model instantiation...")
    
    try:
        import torch
        from model import IntentClassifier, FastIntentClassifier
        from config import MODEL_CONFIG
        
        # Test BiLSTM model
        model = IntentClassifier(
            vocab_size=1000,
            embedding_dim=MODEL_CONFIG['embedding_dim'],
            hidden_dim=MODEL_CONFIG['hidden_dim'],
            num_intents=10,
            dropout=MODEL_CONFIG['dropout'],
            num_layers=MODEL_CONFIG['num_layers']
        )
        
        # Test forward pass
        dummy_input = torch.randint(0, 1000, (2, 30))
        output, attention = model(dummy_input)
        
        print(f"✓ BiLSTM model instantiated successfully")
        print(f"  - Input shape: {dummy_input.shape}")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test CNN model
        fast_model = FastIntentClassifier(
            vocab_size=1000,
            embedding_dim=MODEL_CONFIG['embedding_dim'],
            num_filters=MODEL_CONFIG['num_filters'],
            filter_sizes=MODEL_CONFIG['filter_sizes'],
            num_intents=10,
            dropout=MODEL_CONFIG['dropout']
        )
        
        output = fast_model(dummy_input)
        print(f"✓ CNN model instantiated successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Model error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset():
    """Test if dataset can be loaded"""
    print("\nTesting dataset loading...")
    
    try:
        from dataset import IntentDataset
        from config import PATHS
        
        if not PATHS['training_data'].exists():
            print("✗ Training data not found, skipping dataset test")
            return False
        
        dataset = IntentDataset(PATHS['training_data'], is_training=True)
        
        print(f"✓ Dataset loaded successfully")
        print(f"  - Total samples: {len(dataset)}")
        print(f"  - Vocabulary size: {len(dataset.vocab)}")
        
        # Test getting a sample
        sample = dataset[0]
        print(f"  - Sample text: {sample['text']}")
        print(f"  - Input shape: {sample['input_ids'].shape}")
        print(f"  - Label: {sample['labels'].item()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_directories():
    """Test if required directories exist"""
    print("\nTesting directory structure...")
    
    try:
        from config import PATHS
        
        # Check data directory
        if PATHS['training_data'].parent.exists():
            print(f"✓ Data directory exists: {PATHS['training_data'].parent}")
        else:
            print(f"✗ Data directory missing: {PATHS['training_data'].parent}")
            return False
        
        # Check models directory
        if PATHS['model_checkpoint'].parent.exists():
            print(f"✓ Models directory exists: {PATHS['model_checkpoint'].parent}")
        else:
            print(f"✗ Models directory missing: {PATHS['model_checkpoint'].parent}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Directory error: {e}")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("NLU Module Setup Verification")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Directories", test_directories),
        ("Data Files", test_data_file),
        ("Model", test_model),
        ("Dataset", test_dataset),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:20s} {status}")
    
    print("="*60)
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ All tests passed! You're ready to train the model.")
        print("\nNext steps:")
        print("  1. Review the training data: data/intents.json")
        print("  2. Start training: python train.py")
        print("  3. Evaluate model: python evaluate.py")
        print("  4. Test inference: python inference.py \"your text here\"")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above before training.")
        print("\nCommon fixes:")
        print("  - Install missing packages: pip install -r requirements.txt")
        print("  - Ensure intents.json is in data/ folder")
        print("  - Check file paths in config.py")
    
    print("="*60)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
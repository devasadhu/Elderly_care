# NLU Module - Natural Language Understanding for HeyCare

**Focus**: Intent classification, dialog management, cognitive exercises, and context handling

## 🎯 Objectives

1. **NLU Model Enhancement**: Achieve >95% accuracy for intent classification
2. **Rasa Dialog Management**: Build multi-turn conversation flows
3. **Cognitive Exercise Modules**: Develop memory games and brain training
4. **Context Management**: Track user preferences and conversation history

## 📁 Module Structure

```
nlu/
├── intent_classifier/       # PyTorch intent classification model
├── data/                    # Training data and augmentation
├── dialog_management/       # Rasa conversation flows
├── cognitive_exercises/     # Brain training games
├── context_management/      # User context and preferences
├── models/                  # Saved model checkpoints
└── tests/                   # Unit tests
```

## 🚀 Quick Start

### 1. Prepare Training Data
```bash
cd nlu/data
python synthetic_generator.py  # Generate augmented training data
```

### 2. Train Intent Classifier
```bash
cd nlu/intent_classifier
python train.py
```

### 3. Evaluate Model
```bash
python evaluate.py
```

### 4. Test Real-time Inference
```bash
python inference.py "remind me to take medicine"
```

## 🧠 Intent Categories

### Core Intents (High Priority)
1. **Emergency** - "call for help", "I've fallen"
2. **CallContact** - "call my daughter", "phone John"
3. **SetReminder** - "remind me to take pills at 9 AM"
4. **GetWeather** - "what's the weather today"
5. **SetAlarm** - "set alarm for 7 AM"
6. **GetTime** - "what time is it"

### Secondary Intents
7. **GetNews** - "tell me the news"
8. **TellJoke** - "tell me a joke"
9. **PlayGame** - "let's play a memory game"
10. **GeneralChat** - Casual conversation

### Edge Cases
- **Clarification** - When user input is ambiguous
- **Fallback** - Unrecognized intents
- **CancelAction** - "nevermind", "cancel that"

## 📊 Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Intent Accuracy | >95% | TBD |
| Response Time | <200ms | TBD |
| Elderly Speech Handling | >90% | TBD |

## 🔧 Training Data Requirements

- **Minimum samples per intent**: 100
- **Diversity**: Multiple phrasings, elderly speech patterns
- **Augmentation**: Synthetic data for edge cases
- **Edge case coverage**: Accents, slow speech, hesitations

## 🧪 Testing Strategy

1. **Unit Tests**: Individual component testing
2. **Integration Tests**: End-to-end pipeline
3. **User Testing**: Real elderly user feedback
4. **Edge Case Testing**: Accent variations, background noise

## 📈 Development Roadmap

### Week 1-2: Intent Classifier Foundation
- [x] Set up PyTorch model architecture
- [ ] Create initial training dataset
- [ ] Train baseline model
- [ ] Achieve 85% accuracy

### Week 3-4: Data Enhancement
- [ ] Generate synthetic training data
- [ ] Add elderly speech patterns
- [ ] Expand dataset to 100+ samples per intent
- [ ] Achieve >95% accuracy

### Week 5-6: Rasa Integration
- [ ] Set up Rasa domain and config
- [ ] Create conversation flows
- [ ] Implement custom actions
- [ ] Test multi-turn dialogs

### Week 7-8: Cognitive Exercises
- [ ] Develop memory games
- [ ] Implement adaptive difficulty
- [ ] Add progress tracking

### Week 9: Context Management
- [ ] User preference learning
- [ ] Conversation history
- [ ] Context switching

## 🤝 Integration Points

- **Backend (Krish)**: 
  - Reminder system integration
  - Emergency contact triggers
  - Database queries for user profiles

- **Integration (Deeraj)**:
  - STT → NLU pipeline
  - NLU → TTS response generation
  - End-to-end testing

## 📝 Notes

- All models run on-device for privacy
- Focus on low-latency inference (<200ms)
- Handle diverse elderly speech patterns
- Graceful fallback for unrecognized intents
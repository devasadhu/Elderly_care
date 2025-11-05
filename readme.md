# Elderly_care- Voice-Based Virtual Assistant for Elderly Care

## 🎯 Vision
To enhance the quality of life, safety, and independence of elderly individuals living alone through a proactive and intuitive voice-based virtual assistant.

## 📋 Core Objectives
- **Health & Wellness**: Manage medication schedules, monitor health queries, and provide timely reminders
- **Emergency Assistance**: Offer reliable, hands-free way to call for help in emergencies, including fall detection
- **Social & Cognitive Engagement**: Combat loneliness through communication and cognitive exercises
- **Daily Assistance**: Simplify daily tasks like setting alarms, checking weather, and controlling smart home devices

## 🏗️ Architecture

### Tech Stack
- **Voice Input**: Microphone Array with noise cancellation
- **Wake Word**: On-device TensorFlow Lite model
- **STT**: Google Cloud Speech-to-Text API / WhisperCPP
- **NLU**: Custom PyTorch model + Rasa for dialog management
- **TTS**: Google Cloud Text-to-Speech (WaveNet voices)
- **Integrations**: REST APIs for weather, system calls for contacts

## 👥 Team Structure

### Sadhana - NLU, Dialog Management & AI/ML (35%)
- NLU model enhancement (>95% accuracy)
- Rasa dialog management
- Cognitive exercise modules
- Conversation context management

### Krish - Core Features & Backend (35%)
- Speech pipeline completion
- Reminder & scheduling system
- Emergency contact system
- News briefing feature
- Database & data management

### Deeraj - Integration, UX & Testing (30%)
- Social engagement features
- Voice response & UX refinement
- Weather & daily utilities
- User testing coordination
- Documentation & QA

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/devasadhu/Elderly_care.git
cd Elderly_care
pip install -r requirements.txt
```

### Download Required Models
```bash
# Download spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Run NLU Training
```bash
cd nlu/intent_classifier
python train.py
```

### Test Intent Classification
```bash
python inference.py "HeyCare, remind me to take medicine at 9 AM"
```

## 📊 Project Status

- ✅ **Wake Word Detection**: Completed
- 🔄 **Speech-to-Text**: In Progress
- 🔄 **NLU**: In Progress (Intent model deployed)
- ✅ **Text-to-Speech**: Completed

## 🎯 Current Features

1. **Medication & Appointment Reminders**
   - Voice-activated reminder setting
   - Verbal alerts at scheduled times

2. **Emergency Contact**
   - "HeyCare, call for help" triggers immediate contact

3. **Fall Detection Integration** (In Progress)
   - BLE data link with wearable accelerometer
   - Algorithm refinement ongoing

## 📅 Development Phases

### Phase 1: Model & Algorithm Refinement (Current)
- [ ] NLU accuracy >95%
- [ ] Complete fall detection algorithm
- [ ] Daily news briefing integration

### Phase 2: Feature Expansion
- [ ] Social engagement modules
- [ ] Smart home controls

### Phase 3: User Testing & Feedback
- [ ] Deploy prototype to test group
- [ ] Gather usability feedback
- [ ] Iterate based on user experience

## 📁 Repository Structure

```
Elderly_care/
├── nlu/                    # Sadhana's work - NLU & AI/ML
├── backend/                # Krish's work - Core features
├── integration/            # Deeraj's work - Integration & testing
└── docs/                   # Documentation
```

## 🤝 Contributing

Each team member works on their designated branch:
- `sadhana/nlu-development`
- `krish/backend-development`
- `deeraj/integration-development`

## 📄 License

This project is developed as part of an academic/research initiative for elderly care assistance.

## 📧 Contact

For questions or collaboration inquiries, please reach out to the development team.
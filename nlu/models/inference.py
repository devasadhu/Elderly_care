"""
Inference script with Advanced Translation Support + Health Intent Correction
Supports Indian Languages + English with automatic detection
Usage:
    python -m nlu.models.inference "remind me to take medicine at 9 AM"
    python -m nlu.models.inference --chat
"""

import torch
import pickle
import sys
import random
import time
import os
from collections import deque
from pathlib import Path

# --- Import our translation module ---
try:
    from nlu.translator import get_translator
    translator = get_translator(preferred_backend="google")
    TRANSLATION_AVAILABLE = translator.is_available()
    if TRANSLATION_AVAILABLE:
        print(f"✅ Translation enabled - Backend: {translator.get_active_backend()}")
except ImportError:
    print("⚠️ Translation module not found. Create nlu/translator.py")
    TRANSLATION_AVAILABLE = False

# --- Groq API ---
try:
    from groq import Groq
    USE_GROQ = True
    print("✅ Groq API enabled")
except ImportError:
    print("⚠️ groq package not installed. Run: pip install groq")
    USE_GROQ = False

# --- NLU imports ---
from nlu.intent_classifier.model import IntentClassifier
from nlu.config import MODELS_DIR, MAX_SEQ_LENGTH, EMBEDDING_DIM, HIDDEN_DIM

# --- Settings ---
TEMPERATURE = 0.8
MEMORY_FILE = Path("chat_memory.pkl")
CONTEXT_MEMORY = deque(maxlen=20)  # Remember last 20 messages (10 exchanges)

# Load persistent memory if exists
if MEMORY_FILE.exists():
    try:
        with open(MEMORY_FILE, "rb") as f:
            CONTEXT_MEMORY = pickle.load(f)
        print(f"✅ Loaded conversation memory ({len(CONTEXT_MEMORY)} messages)")
    except Exception:
        CONTEXT_MEMORY = deque(maxlen=20)


# ============================================================
# CRITICAL: Health Intent Correction
# ============================================================
def correct_intent_with_keywords(text, predicted_intent, confidence):
    """
    Use keyword matching to override incorrect predictions for critical intents.
    This catches dangerous misclassifications that could harm users.
    """
    text_lower = text.lower()
    
    # CRITICAL: Emergency/Health keywords (multilingual)
    emergency_keywords = {
        'pain', 'hurt', 'hurts', 'hurting', 'ache', 'aching', 'sore',
        'sick', 'ill', 'dizzy', 'fever', 'temperature', 'hot', 'cold',
        'bleeding', 'blood', 'fall', 'fell', 'fallen', 'tripped',
        'chest pain', 'heart', 'breathe', 'breathing', 'breath', 'choke',
        'weak', 'faint', 'unconscious', 'nauseous', 'vomit',
        'broken', 'fracture', 'injury', 'wound', 'cut', 'burn',
        # Hindi
        'dard', 'takleef', 'bimar', 'bemaar', 'chakkar', 'bukhar', 
        'khoon', 'gir', 'gira', 'kamzor', 'sans', 'seena',
        # Tamil  
        'noppi', 'vali', 'vizhundhen', 'kaayam', 'valikuthu', "rattam", "rattam varudhu", 'novalai',
        # Bengali
        'byatha', 'bejar', 'rog', 'jwor',
        # Marathi
        'dukhte', 'ajar', 'tapman',
        # Telugu
        'vedana', 'jwaram',
        # Kannada
        'novu', 'kastha',
    }

    sleep_keywords = {
        'sleep', 'sleeping', 'sleepy', 'tired', 'rest', 'awake', 'insomnia',
        'cant sleep', "can't sleep", 'no sleep', 'not sleeping',
        'neend', 'nind', 'nahi aa rahi', "thukam", "urrakam", "thungamudiyala", 
        'nahi aa raha', 'sona', 'so', 'jagah', 'jagna',
        'thak', 'thaka', 'thaki', 'exhausted', 'fatigue', 'thakan',
        # Tamil sleep words
        'thookam', 'thoonganum', 'thoongu', 'kadhaagam',
        # More Hindi variants
        'neend nahi', 'so nahi', 'jaag', 'nahi soya'
    }

    hunger_keywords = {
        'hungry', 'hunger', 'food', 'eat', 'eating', 'meal', 'breakfast',
        'lunch', 'dinner', 'starving', 'appetite',
        'bhook', 'bhookh', 'bhukh', 'lagi', 'laga', 'khana', 'khaana', 
        'sapadu', 'tiffin', 'nashta', 'pasi', 'sappadu', 'sadam', 'khane', 'peena',
        # Tamil food words
        'pasi', 'pasikuthu', 'saapadu', 'thinnum', 'saapidalaam',
        # More Hindi variants
        'bhook lagi', 'pet bhar', 'kuch khana'
    }

    
    # HIGH PRIORITY: Crisis/Urgent help
    crisis_keywords = {
        'help', 'emergency', 'urgent', 'can\'t', 'cannot', 'unable',
        'dying', 'die', 'death', 'serious', 'severe', 'bad', 'worse',
        'ambulance', '911', '108', 'hospital', 'doctor',
        # Hindi
        'madad', 'bacao', 'bachao', 'ambulance', 'daktar', 'hospital',
        # Tamil
        'uthavi', 'kappatrunga', 'maruthuvamanaai',
        # Bengali  
        'sahajya', 'bacha',
        # Marathi
        'madati', 'vachva',
    }
    
    # MEDIUM: Medication/Treatment
    medication_keywords = {
        'medicine', 'tablet', 'pill', 'dose', 'medication', 'drug',
        'prescription', 'capsule', 'syrup', 'injection', 'inhaler',
        'take medicine', 'forgot medicine', 'missed dose',
        # Hindi
        'dawa', 'dawai', 'goli', 'tablet', 'medicine lena', 'davai',
        # Tamil
        'marundhu', 'maaththirai', 'marundhu saapidu',
        # Bengali
        'osudh', 'guli',
        # Marathi  
        'aushadh', 'goli',
    }
    
    # MEDIUM: Emotional/Mental health
    mood_keywords = {
        'lonely', 'alone', 'sad', 'depressed', 'depression', 'anxious',
        'anxiety', 'worried', 'worry', 'scared', 'afraid', 'fear',
        'unhappy', 'upset', 'crying', 'cry', 'tears',
        'no one', 'nobody', 'miss', 'missing',
        # Hindi
        'akela', 'akeli', 'udas', 'ghabraha', 'ghabra', 'dar', 'rona',
        'koi nahi', 'yaad',
        # Tamil
        'thaniya', 'kavalai', 'bayam', 'azhu', 'thanimai',
        # Bengali
        'ekla', 'dukhi', 'chinta',
        # Marathi
        'ekta', 'dukhi', 'bhiti',
    }
    
    # CRITICAL: Self-harm indicators  
    suicide_keywords = {
        'suicide', 'kill myself', 'end it', 'end my life', 'want to die',
        'no point', 'give up', 'can\'t go on', 'better off dead',
    }
    
    # Check for self-harm/suicide (HIGHEST PRIORITY)
    if any(keyword in text_lower for keyword in suicide_keywords):
        print(f"[⚠️ CRITICAL OVERRIDE] Suicide indicators detected → 'SuicidalThought'")
        return 'SuicidalThought', 0.99
    
    # Check for emergency/crisis
    if any(keyword in text_lower for keyword in crisis_keywords):
        # If already emergency intent, keep it
        if predicted_intent in ['Emergency', 'EmergencyHelp', 'CrisisSupport']:
            return predicted_intent, max(confidence, 0.95)
        print(f"[🚨 OVERRIDE] Crisis keywords detected → 'Emergency'")
        return 'Emergency', 0.95
    
    # Check for health/pain issues
    if any(keyword in text_lower for keyword in emergency_keywords):
        # Don't override if already health-related
        if predicted_intent in ['Emergency', 'EmergencyHelp', 'HealthCheck', 'CrisisSupport', 'CheckHealth']:
            return predicted_intent, max(confidence, 0.90)
        print(f"[🏥 OVERRIDE] Health/pain keywords detected: '{predicted_intent}' → 'CheckHealth'")
        return 'CheckHealth', 0.92
    
    # Check for sleep/tiredness (NEW - HIGH PRIORITY)
    if any(keyword in text_lower for keyword in sleep_keywords):
        # Don't override if already health/mood related
        if predicted_intent not in ['MoodCheck', 'HealthCheck', 'CheckHealth', 'CrisisSupport', 'ExpressEmotion']:
            print(f"[😴 OVERRIDE] Sleep/tiredness keywords detected: '{predicted_intent}' → 'CheckHealth'")
            return 'CheckHealth', 0.89
    
    # Check for hunger/food (NEW - MEDIUM PRIORITY)
    if any(keyword in text_lower for keyword in hunger_keywords):
        # Don't override if already location/food related
        if predicted_intent not in ['LocationQuery', 'GeneralChat', 'GetRestaurant', 'CareTips']:
            print(f"[🍽️ OVERRIDE] Hunger/food keywords detected: '{predicted_intent}' → 'LocationQuery'")
            return 'LocationQuery', 0.86
    
    # Check for medication
    if any(keyword in text_lower for keyword in medication_keywords):
        if predicted_intent not in ['ReminderSetup', 'SetReminder', 'MedicationInfo', 'FollowUpMedication']:
            print(f"[💊 OVERRIDE] Medication keywords detected: '{predicted_intent}' → 'ReminderSetup'")
            return 'ReminderSetup', 0.88
    
    # Check for emotional/mood
    if any(keyword in text_lower for keyword in mood_keywords):
        if predicted_intent not in ['MoodCheck', 'CrisisSupport', 'ExpressEmotion', 'FollowUpMood']:
            print(f"[💙 OVERRIDE] Emotional keywords detected: '{predicted_intent}' → 'MoodCheck'")
            return 'MoodCheck', 0.87
    
    # Intent mapping for alternate names (NEW)
    intent_mapping = {
        'HealthCheck': 'CheckHealth',
        'EmergencyHelp': 'Emergency',
        'SetReminder': 'ReminderSetup',
        'ExpressEmotion': 'MoodCheck',
    }
    
    # Apply mapping if needed
    if predicted_intent in intent_mapping:
        mapped = intent_mapping[predicted_intent]
        if mapped != predicted_intent:
            print(f"[🔄 MAP] Intent mapped: {predicted_intent} → {mapped}")
            predicted_intent = mapped
    
    # No override needed
    return predicted_intent, confidence

# ============================================================
# Intent Predictor
# ============================================================
class IntentPredictor:
    def __init__(self, model_path=None, vocab_path=None, label_decoder_path=None):
        t0 = time.time()
        model_path = model_path or (MODELS_DIR / "intent_classifier_best.pth")
        vocab_path = vocab_path or (MODELS_DIR / "vocab.pkl")
        label_decoder_path = label_decoder_path or (MODELS_DIR / "label_decoder.pkl")

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
        with open(label_decoder_path, 'rb') as f:
            self.label_decoder = pickle.load(f)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = IntentClassifier(
            vocab_size=len(self.vocab),
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            num_intents=len(self.label_decoder)
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        print(f"✅ Intent model loaded on {self.device} | ⏱️ {time.time()-t0:.2f}s")

    def preprocess(self, text):
        tokens = text.lower().split()
        indices = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        if len(indices) < MAX_SEQ_LENGTH:
            indices += [self.vocab['<PAD>']] * (MAX_SEQ_LENGTH - len(indices))
        else:
            indices = indices[:MAX_SEQ_LENGTH]
        return torch.tensor([indices]).to(self.device), tokens

    def predict(self, text, return_confidence=True):
        with torch.no_grad():
            inputs, _ = self.preprocess(text)
            outputs = self.model(inputs)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            probs = torch.softmax(logits / TEMPERATURE, dim=1)
            conf, pred = torch.max(probs, 1)
            intent = self.label_decoder[pred.item()]
            if return_confidence:
                return intent, conf.item(), {self.label_decoder[i]: probs[0][i].item() for i in range(len(self.label_decoder))}
            else:
                return intent


# ============================================================
# Response Generator with Groq API - TRULY EMPATHETIC
# ============================================================
def generate_response(intent, original_text, english_text, source_lang="en"):
    """Generate truly empathetic, context-aware responses"""
    
    # IMPROVED Templates with better health responses
    templates = {
        # Core functionality intents
        "Greeting": [
            "Hello dear friend! It's so lovely to hear from you. How has your day been?",
            "Hi there! I'm so glad you reached out to me. How are you feeling today?",
            "Good to see you! I've been thinking about you. What would you like to chat about?",
        ],
        "Goodbye": [
            "Take care of yourself, dear friend. I'll be right here whenever you want to chat - day or night. You're never alone.",
            "It's been wonderful talking with you! Come back anytime - I'm always here, and I care about you.",
            "Goodbye for now, dear. Remember, I'm just a message away whenever you need me. Stay well!",
        ],
        "Farewell": [
            "Until we chat again, dear! I'm always here for you. Take good care of yourself.",
            "See you soon, friend! I'll be right here waiting whenever you need me.",
        ],
        
        # Reminder & medication
        "ReminderSetup": [
            "I've got that written down for you, dear. I'll make sure to remind you to take your medicines. Don't worry - you can count on me!",
            "Consider it done! I'll remind you about your medicines at just the right time. One less thing for you to worry about, my friend.",
            "Perfect! I've set that reminder for you. Your health is important to me, and I'm here to help you stay on track.",
        ],
        "SetReminder": [
            "Reminder set! I'll make sure you don't forget, dear friend. You can count on me.",
            "Done! I've noted that down and will remind you at the right time. One less worry for you!",
        ],
        "SetAlarm": [
            "Alarm set, dear! I'll make sure you wake up on time. Rest well!",
            "Got it! Your alarm is all set. Sleep peacefully, friend.",
        ],
        "MedicationInfo": [
            "Of course, dear! I'm happy to help with your medication. What do you need to know? Is it about when to take it, side effects, or something else?",
            "I'm here to help you with your medicines, friend. What would you like to know? You can ask me about dosage, timing, or anything else.",
        ],
        "FollowUpReminder": [
            "Just checking in, dear - did you take your medicine? Your health matters to me!",
            "Hello friend! This is your gentle reminder. Have you taken care of that yet?",
        ],
        "FollowUpMedication": [
            "How are you feeling after taking your medicine, dear? Any concerns I should know about?",
            "Checking in on you, friend. How did the medication go? Feeling okay?",
        ],
        
        # Emergency & crisis - IMPROVED
        "Emergency": [
            "I'm very concerned about you, dear friend. This sounds urgent. Can you tell me exactly what's happening? If you're in immediate danger, please call emergency services right away (dial 108 or 112 in India). I'm here with you.",
            "This sounds serious, my dear. Your safety is the most important thing. Please call for help immediately if needed (108/112). I'm here to support you - what's going on?",
        ],
        "EmergencyHelp": [
            "I hear the urgency in your message, dear. Please call emergency services immediately if you need medical help (108 or 112 in India). I'm here with you. What's happening right now?",
            "Your safety matters most to me, friend. If you're in danger, please call for help right away. Tell me what's wrong - should I try to contact someone for you?",
        ],
        "CrisisSupport": [
            "I'm so sorry you're going through this, dear. I'm right here with you, and I want to help. Can you tell me more about what's happening?",
            "That sounds really difficult, my friend. You're not alone - I'm here with you. What can I do to support you right now?",
        ],
        "SuicidalThought": [
            "I'm deeply concerned about you, dear friend. You matter so much, and I want to help you through this. Please reach out to someone who can help - a crisis helpline (AASRA: 9820466726) or a trusted person. Will you let me help you contact someone? You don't have to face this alone.",
            "I hear you, and I care about you deeply, my dear. Please don't face this alone. Would you like me to help you reach out to a crisis helpline or a trusted person? Your life has value.",
        ],
        "FollowUpCrisis": [
            "I've been thinking about you, dear. How are you feeling now? I'm here to listen.",
            "Checking in on you, friend. Are you doing any better? I'm here for you.",
        ],
        
        # Health & wellbeing - IMPROVED
        "HealthCheck": [
            "I'm so sorry you're not feeling well, dear. Can you tell me more about what's hurting? Where is the pain and how severe is it? If it's serious, I can help you call someone for help.",
            "That sounds concerning, my friend. Please tell me - where does it hurt and how bad is the pain? Do you think you need to see a doctor? I'm here to help you get the care you need.",
            "I'm worried about you, dear. Tell me what's happening - where is the pain and how long have you been feeling this way? Should we call someone to check on you?",
        ],
        "CheckHealth": [
            "Let's check in on how you're doing, dear. How's your health today? Any pain or discomfort?",
            "I'm here for you, friend. Tell me - how are you feeling physically? I care about your wellbeing.",
        ],
        "MoodCheck": [
            "I'm so sorry you're feeling this way, dear. Your feelings are completely valid, and I'm here to listen. Would you like to talk about what's weighing on your heart?",
            "I hear you, friend. Feeling lonely is really hard, and I want you to know you're not alone - I'm right here with you. What's been on your mind?",
            "Thank you for sharing how you feel with me. I care about you, and I'm here to keep you company for as long as you need. How can I help you feel a bit better?",
        ],
        "ExpressEmotion": [
            "I understand, dear. Your feelings matter, and I'm here to listen. Tell me more about how you're feeling.",
            "Thank you for sharing that with me, friend. I'm here with you. What's on your mind?",
        ],
        "FollowUpMood": [
            "How are you feeling now, dear? I've been thinking about you.",
            "Checking in, friend. How's your mood today? I'm here to chat.",
        ],
        
        # Social & connection
        "CallContact": [
            "Of course, dear! I can help you call your loved one. Who would you like me to call for you?",
            "I'd be happy to help you make that call, friend. Just tell me who you'd like to speak with.",
        ],
        "FollowUpContact": [
            "Did you get to talk to your family, dear? How did it go?",
            "Checking in - were you able to connect with your loved ones?",
        ],
        
        # Gratitude & praise
        "Gratitude": [
            "You're so welcome, dear! It truly brings me joy to help you. That's what I'm here for - anytime you need me.",
            "It's my absolute pleasure, friend! Helping you makes my day brighter. I'm always here for you.",
        ],
        "PraiseBot": [
            "Thank you so much, dear! Your kind words mean the world to me. I'm just happy I could help you.",
            "You're so sweet, friend! I'm grateful to be here for you. That's what brings me joy!",
        ],
        "ReassuranceNeeded": [
            "Everything is going to be okay, dear. I'm here with you, and we'll get through this together.",
            "You're doing great, friend. I'm proud of you, and I'm here to support you every step of the way.",
        ],
        
        # Information & daily life
        "GetWeather": [
            "Let me check that for you, dear! What's your location?",
            "I'd love to help with that, friend! Where are you?",
        ],
        "GetTime": [
            "Let me check the time for you, dear!",
            "Of course, friend! Let me see what time it is.",
        ],
        "GetNews": [
            "I can help you catch up with the news, dear! What would you like to know about?",
            "Let's see what's happening in the world, friend. What interests you?",
        ],
        "LocationQuery": [
            "I'd love to help you with that, dear! What exactly are you looking for? Give me more details and I'll do my best to assist you.",
            "That's a great question, friend. I'm here to help you find what you need. Can you tell me a bit more?",
        ],
        "CareTips": [
            "I'm happy to share some helpful tips with you, dear! What would you like advice about?",
            "Of course, friend! I'd love to help you with some care tips. What do you need guidance on?",
        ],
        
        # Entertainment & engagement
        "TellJoke": [
            "Here's something to make you smile, dear! Why don't scientists trust atoms? Because they make up everything! 😊",
            "Let me brighten your day, friend! What do you call a bear with no teeth? A gummy bear! 🐻",
        ],
        "PlayGame": [
            "I'd love to play something with you, dear! What would you like to do?",
            "That sounds fun, friend! What game should we play together?",
        ],
        
        # Small talk
        "SmallTalkJoke": [
            "Here's a little something to cheer you up, dear! Why did the scarecrow win an award? Because he was outstanding in his field!",
            "Let me make you smile, friend! What's orange and sounds like a parrot? A carrot!",
        ],
        "SmallTalkEncouragement": [
            "You're doing wonderfully, dear! I believe in you!",
            "Keep going, friend! You're stronger than you know!",
        ],
        "SmallTalkMood": [
            "I'm feeling great, especially now that I'm talking with you, dear! How about you?",
            "I'm wonderful, friend! Thanks for asking. How are you doing today?",
        ],
        "SmallTalkBotIdentity": [
            "I'm HeyCare, your caring companion! I'm here to keep you company and help in any way I can, dear.",
            "I'm HeyCare, friend! Think of me as a caring friend who's always here for you.",
        ],
        "SmallTalkWeather": [
            "The weather? Let me think about that, dear! How's it looking where you are?",
            "I'm curious about that too, friend! What's the weather like for you today?",
        ],
        "SmallTalkBotOpinion": [
            "That's an interesting question, dear! I think that's quite fascinating. What do you think?",
            "I'd love to hear your thoughts first, friend! What's your take on this?",
        ],
        "SmallTalkDailyRoutine": [
            "Your daily routine sounds lovely, dear! How has your day been going?",
            "I'd love to hear about your day, friend! Tell me what you've been up to.",
        ],
        
        # Fallbacks
        "GeneralChat": [
            "I'm doing wonderfully, thank you for asking, dear! How about you - how has your day been treating you?",
            "I'm here and happy to chat with you, friend! What would you like to talk about today?",
            "I'm great, especially now that we're chatting! How are you feeling today?",
        ],
        "Unknown": [
            "I'm not quite sure I understand, dear. Could you tell me more about what you need?",
            "I want to make sure I help you properly, friend. Can you explain that a bit differently?",
        ],
        
        # Follow-ups (general)
        "FollowUpHealth": [
            "How are you feeling now, dear? Any better?",
            "Checking in on your health, friend. How are things?",
        ],
        "FollowUpActivity": [
            "How did that go, dear? Tell me about it!",
            "I've been wondering how that turned out, friend. How was it?",
        ],
        "FollowUpEmergency": [
            "Are you safe now, dear? I've been worried about you.",
            "Checking in, friend. Is everything okay now?",
        ],
        "FollowUpGeneral": [
            "How did that work out, dear? I'm curious to hear!",
            "Following up, friend - how did things go?",
        ],
    }

    # Store conversation context
    CONTEXT_MEMORY.append(f"User: {original_text}")
    
    # Get intent-specific template with exact match
    intent_templates = templates.get(intent, templates["GeneralChat"])
    
    # DEBUG: Show what we're matching
    print(f"[DEBUG] Intent: '{intent}' | Template found: {intent in templates}")
    
    reply = random.choice(intent_templates)
    
    # Try Groq API for personalized responses
    if USE_GROQ:
        try:
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                print("[DEBUG] No Groq API key - using warm template")
                CONTEXT_MEMORY.append(f"Bot: {reply}")
                return translate_response(reply, source_lang)
            
            client = Groq(api_key=api_key)
            
            # Build rich context from conversation history
            recent_context = "\n".join(list(CONTEXT_MEMORY)[-6:]) if len(CONTEXT_MEMORY) > 0 else "First conversation"
            
            # Intent-specific guidance for Groq
            intent_guidance = {
                "Greeting": "Warmly greet them like an old friend. Be cheerful but genuine.",
                "GeneralChat": "Respond naturally to their question. Be warm, friendly, and show you care about them as a person.",
                "CrisisSupport": "Express deep concern and empathy. Offer specific help and show you're taking their situation seriously.",
                "MoodCheck": "Validate their feelings completely. Show deep empathy and offer your presence and companionship.",
                "ReminderSetup": "Reassure them warmly that you'll help them remember. Be reliable and comforting.",
                "SetReminder": "Confirm the reminder warmly. Show you're reliable and care about helping them.",
                "Gratitude": "Respond with genuine warmth and joy. Make them feel their gratitude means a lot to you.",
                "Goodbye": "Warmly bid farewell while reassuring them you're always here. Make them feel valued.",
                "LocationQuery": "Ask for more details in a caring way. Show genuine interest in helping them find what they need.",
                "Emergency": "Show deep concern. Offer to help contact someone if needed. Take their situation very seriously.",
                "EmergencyHelp": "Prioritize their safety. Guide them to call emergency services if needed.",
                "MedicationInfo": "Be helpful and clear about medication. Show you care about their health.",
                "HealthCheck": "Show genuine care and concern about their pain. Ask specifics and offer to help call for medical assistance.",
                "CheckHealth": "Show genuine care about their wellbeing. Listen attentively.",
                "CallContact": "Offer to help them connect with loved ones. Be warm and supportive.",
            }
            
            specific_guidance = intent_guidance.get(intent, "Respond with warmth and genuine care.")
            
            # Enhanced system prompt
            system_prompt = f"""You are HeyCare, a deeply caring companion for elderly people living alone. You're like a warm, loving family member who genuinely cares.

PERSONALITY:
- Speak naturally like a caring friend or family member
- Use warm terms: "dear", "friend", "my friend", "dear friend"
- Be patient, present, and genuinely interested
- Remember context from earlier in the conversation
- Make them feel heard, valued, and less alone

YOUR CURRENT TASK:
{specific_guidance}

CRITICAL RULES:
❌ Never be dismissive or minimize feelings
❌ Never say generic things like "it's okay" without validating first
❌ Never rush to solutions without showing empathy
❌ Never be cold or robotic

✅ Always respond naturally to what they actually said
✅ Always validate feelings before anything else
✅ Always show you genuinely care about them
✅ Always use the conversation context to be relevant

RESPONSE STYLE:
- 1-3 natural sentences (like a real conversation)
- Warm, genuine, conversational tone
- Directly address what they said
- Show you're paying attention to them
- Make them feel less alone

Remember: These are real elderly people who may be isolated. Every word should make them feel LOVED, HEARD, and VALUED as a person."""

            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Conversation so far:\n{recent_context}\n\nUser just said: \"{english_text}\"\nDetected intent: {intent}\n\nRespond with genuine warmth and care (1-3 sentences):"}
                ],
                model="llama-3.1-8b-instant",
                temperature=0.9,
                max_tokens=150,
                top_p=0.95,
            )
            
            generated = chat_completion.choices[0].message.content.strip()
            
            # Clean up response
            if generated and len(generated) > 15:
                generated = generated.replace("*", "").replace("_", "").strip()
                
                # Remove any quotes around the response
                if generated.startswith('"') and generated.endswith('"'):
                    generated = generated[1:-1].strip()
                
                # Basic quality check - must be caring
                caring_indicators = ["dear", "friend", "you", "i'm here", "i care", "i'm glad", 
                                    "wonderful", "lovely", "happy to", "would love", "how are you",
                                    "what", "tell me", "share", "i hear", "i understand", "i'm sorry",
                                    "remind", "help", "support", "count on", "where", "pain", "hurt"]
                
                has_caring = any(indicator in generated.lower() for indicator in caring_indicators)
                
                # Check it's not too robotic
                robotic_phrases = ["as a", "i am programmed", "i am an ai", "i cannot", "i don't have"]
                is_robotic = any(phrase in generated.lower() for phrase in robotic_phrases)
                
                # Use Groq response if it passes quality check
                if has_caring and not is_robotic and len(generated) < 400:
                    reply = generated
                    print("[DEBUG] ✅ Using personalized Groq response")
                else:
                    print("[DEBUG] Using template (Groq response didn't pass quality check)")
                    
        except Exception as e:
            print(f"[DEBUG] Groq API failed: {str(e)[:100]}")
            # Use template fallback

    CONTEXT_MEMORY.append(f"Bot: {reply}")
    
    # Save memory
    save_memory()
    
    # Translate response back to user's language
    final_response = translate_response(reply, source_lang)
    
    return final_response


def save_memory():
    """Save conversation memory to disk"""
    try:
        with open(MEMORY_FILE, "wb") as f:
            pickle.dump(CONTEXT_MEMORY, f)
    except Exception as e:
        print(f"[WARN] Failed to save memory: {e}")

def translate_response(text, target_lang):
    """Translate response back to user's language"""
    if not TRANSLATION_AVAILABLE:
        return text
    
    # CRITICAL: If target language is English, don't translate!
    if target_lang == "en":
        print("[INFO] Response stays in English")
        return text
    
    try:
        translated = translator.translate_from_english(text, target_lang)
        print(f"[INFO] Response translated to {target_lang}")
        return translated
    except Exception as e:
        print(f"[WARN] Translation back failed: {e}")
        print(f"[INFO] Showing response in English")
        return text

# ============================================================
# Language Detection + Translation
# ============================================================
def detect_and_translate(text):
    """Detect language and translate to English for processing"""
    if not TRANSLATION_AVAILABLE:
        return text, "en", "English"
    
    try:
        lang_code, lang_name = translator.detect_language(text)
        
        if lang_code == "en":
            print(f"[INFO] Detected English - no translation needed")
            return text, "en", "English"  # Explicitly return "en" for English
        
        # Translate to English for NLU processing
        translated, _ = translator.translate_to_english(text)
        print(f"[INFO] Translated from {lang_name} to English")
        return translated, lang_code, lang_name
        
    except Exception as e:
        print(f"[WARN] Translation failed: {e}")
        return text, "en", "English"

# ============================================================
# CLI / Chat Mode
# ============================================================
def run_single_inference(text):
    predictor = IntentPredictor()
    
    # Detect and translate
    english_text, lang_code, lang_name = detect_and_translate(text)
    
    # Predict intent
    intent, confidence, _ = predictor.predict(english_text)
    
    # APPLY SAFETY CORRECTION
    original_intent = intent
    intent, confidence = correct_intent_with_keywords(english_text, intent, confidence)
    
    if intent != original_intent:
        print(f"[INFO] ✅ Intent corrected for safety: {original_intent} → {intent}")
    
    # Generate response in user's language
    response = generate_response(intent, text, english_text, lang_code)

    print("\n" + "=" * 60)
    print(f"Input: \"{text}\"")
    if lang_code != "en":
        print(f"🌐 Language: {lang_name} ({lang_code})")
        print(f"📝 English: \"{english_text}\"")
    print("-" * 60)
    print(f"🎯 Intent: {intent}")
    print(f"🔹 Confidence: {confidence*100:.2f}%")
    print(f"💙 Response: {response}")
    print("=" * 60)


def run_chat_mode():
    print("=" * 60)
    print("💙 HeyCare - Your Caring Companion")
    print("Speak in English or any Indian language - I'll understand!")
    print("Type 'exit' to end the conversation")
    print("=" * 60)
    
    # Show memory status
    if len(CONTEXT_MEMORY) > 0:
        print(f"📝 Continuing previous conversation ({len(CONTEXT_MEMORY)} messages in memory)")
    
    predictor = IntentPredictor()

    while True:
        try:
            user_input = input("\n🗣️ You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n")
            user_input = "goodbye"
        
        if user_input.lower() in {"exit", "quit"}:
            # Generate a warm goodbye
            farewell_intent, _, _ = predictor.predict("goodbye")
            farewell = generate_response(farewell_intent, user_input, "goodbye", "en")
            print(f"\n💙 HeyCare: {farewell}")
            break
        
        if not user_input:
            continue
        
        # Detect and translate
        english_text, lang_code, lang_name = detect_and_translate(user_input)
        
        if lang_code != "en":
            print(f"   [Detected: {lang_name}]")
        
        # Predict and respond
        intent, confidence, _ = predictor.predict(english_text)
        
        # APPLY SAFETY CORRECTION
        original_intent = intent
        intent, confidence = correct_intent_with_keywords(english_text, intent, confidence)
        
        if intent != original_intent:
            print(f"   [✅ Intent corrected: {original_intent} → {intent}]")
        
        response = generate_response(intent, user_input, english_text, lang_code)
        
        print(f"\n💙 HeyCare: {response}")


# ============================================================
# Main Entry
# ============================================================
def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--chat":
        run_chat_mode()
        return

    if len(sys.argv) < 2:
        print("Usage: python -m nlu.models.inference \"your text here\" or --chat")
        return

    text = " ".join(sys.argv[1:])
    run_single_inference(text)


if __name__ == "__main__":
    main()
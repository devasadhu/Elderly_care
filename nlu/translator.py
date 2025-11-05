"""
Advanced Translation Module for Elderly Care Assistant
Supports Indian Languages + English
"""

import os
from pathlib import Path
from typing import Tuple, Optional
import json
import re

# ============================================================
# Translation Backends (Multiple Options)
# ============================================================

class TranslationBackend:
    """Base class for translation backends"""
    
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        raise NotImplementedError
    
    def detect_language(self, text: str) -> str:
        raise NotImplementedError


# ============================================================
# Google Translator Backend (Free, No API Key)
# ============================================================
class GoogleTranslatorBackend(TranslationBackend):
    """Using deep-translator (free, no API key needed)"""
    
    def __init__(self):
        try:
            from deep_translator import GoogleTranslator
            self.GoogleTranslator = GoogleTranslator
            self.available = True
            print("✅ Google Translator (Free) loaded")
        except ImportError:
            self.available = False
            print("⚠️ deep-translator not installed. Run: pip install deep-translator")
    
    def detect_language(self, text: str) -> str:
        """Detect language with focus on Indian languages + English"""
        if not self.available:
            return "en"
        
        try:
            # Step 1: Check for Indian scripts (Unicode ranges)
            has_devanagari = any('\u0900' <= char <= '\u097F' for char in text)
            if has_devanagari:
                return "hi"  # Hindi/Marathi (default to Hindi)
            
            has_tamil = any('\u0B80' <= char <= '\u0BFF' for char in text)
            if has_tamil:
                return "ta"
            
            has_telugu = any('\u0C00' <= char <= '\u0C7F' for char in text)
            if has_telugu:
                return "te"
            
            has_bengali = any('\u0980' <= char <= '\u09FF' for char in text)
            if has_bengali:
                return "bn"
            
            has_gujarati = any('\u0A80' <= char <= '\u0AFF' for char in text)
            if has_gujarati:
                return "gu"
            
            has_kannada = any('\u0C80' <= char <= '\u0CFF' for char in text)
            if has_kannada:
                return "kn"
            
            has_malayalam = any('\u0D00' <= char <= '\u0D7F' for char in text)
            if has_malayalam:
                return "ml"
            
            has_punjabi = any('\u0A00' <= char <= '\u0A7F' for char in text)
            if has_punjabi:
                return "pa"
            
            # Step 2: Check for Romanized text (Latin script only)
            is_ascii = all(ord(char) < 128 or char.isspace() for char in text)
            
            if is_ascii:
                text_lower = text.lower()
                
                # PRIORITY 1: Check Tamil FIRST (most specific)
                tamil_keywords = [
                    # Pronouns and basic words
                    "naan", "naanu", "enakku", "enaaku", "enak", 
                    # Greetings and questions
                    "vanakkam", "epadi", "eppadi", "enna", "yenna", "eppo", "yen",
                    # Verbs (common Tamil verb forms)
                    "irukku", "iruku", "iruken", "irukeyn", "pannanum", "pannu", "panna",
                    "ponga", "vanga", "sollu", "sollunga", "vaanga", "paren", "paakalam",
                    "seiyanum", "seyya", "kudukanum", "kudu", "varum", "pogum",
                    # Family and people
                    "paiyan", "paiyanukku", "ponnu", "amma", "appa", "thatha", "paati",
                    # Common words
                    "vali", "valikuthu", "venum", "vena", "vendam", "vendum",
                    "ungaluku", "ungal", "romba", "nalla", "kettadhu", "nandri",
                    "sapadu", "thanni", "veetla", "kelamba",
                    # Food and daily life
                    "saapadu", "thinnu", "kudikanum", "thunganum", "nalla"
                ]
                tamil_match_count = sum(1 for word in tamil_keywords if word in text_lower)
                
                if tamil_match_count >= 1:
                    return "ta"
                
                # PRIORITY 2: Check Hindi (with improved keywords)
                hindi_keywords = [
                    # Core Hindi words
                    "mujhe", "mujhko", "hai", "hoon", "hain", "tha", "thi", "the",
                    "mera", "meri", "mere", "tumhara", "tumhari", "tumhare",
                    "kya", "kaise", "kaisa", "kaisi", "kyu", "kyun", "kyon",
                    "nahi", "nahin", "mat", "bilkul", "nai",
                    "aap", "aapka", "tum", "tumko", "main", "hum", "humko",
                    # Health and body
                    "dard", "takleef", "madad", "dowaai", "dawaai", "dawa", "dawai",
                    "neend", "nind", "sona", "so", "jagah", "jagna",
                    # Food and needs
                    "bhook", "bhookh", "pyaas", "pyaasa", "khana", "khaana", "peena",
                    # States and actions
                    "thak", "thaka", "thaki", "gaya", "gayi", "gaye", "raha", "rahi", "rahe",
                    "bahut", "bohot", "thoda", "zara", "bhi", "bheee",
                    # Question words and connectors
                    "kuch", "koi", "kisi", "yeh", "ye", "woh", "wo", "wahi",
                    "kab", "kahan", "kahaan", "kaise", "kaisay", "kyun", "kaise",
                    "abhi", "abhee", "phir", "fir", "aur", "ya", "yaa",
                    "lekin", "par", "magar", "per",
                    # Common words
                    "thik", "theek", "achha", "acha", "accha", "bura", "bada", "chota",
                    "lagta", "lagi", "laga", "lage", "aa", "aana", "jana", "jao"
                ]
                
                hindi_matches = sum(1 for word in hindi_keywords if word in text_lower)
                
                # Hindi detection logic (need at least 2 matches OR 1 match in short text)
                if hindi_matches >= 2:
                    return "hi"
                
                if hindi_matches >= 1 and len(text.split()) <= 5:
                    return "hi"
                
                # PRIORITY 3: Check Telugu
                telugu_keywords = [
                    "nenu", "naaku", "ela", "emiti", "enduku", "cheppu", 
                    "undi", "unnaru", "unnanu", "ledu", "kavali", "vastunna",
                    "chaala", "baadhaga", "baaga", "annam", "neeru"
                ]
                telugu_match_count = sum(1 for word in telugu_keywords if word in text_lower)
                
                if telugu_match_count >= 1:
                    return "te"
                
                # PRIORITY 4: Check Bengali
                bengali_keywords = ["ami", "amar", "kemon", "ki", "keno", "tomake", "tomar"]
                if any(word in text_lower for word in bengali_keywords):
                    return "bn"
                
                # Step 3: Check if it's English
                english_indicators = [
                    "i", "am", "is", "are", "the", "a", "an", "my", "you", "your",
                    "feel", "feeling", "want", "need", "help", "please", "thank",
                    "hello", "hi", "bye", "goodbye", "how", "what", "where", "when",
                    "lonely", "pain", "tired", "hungry", "thirsty", "sad", "happy",
                    "can", "could", "would", "should", "me", "him", "her"
                ]
                
                words = text_lower.split()
                english_word_count = sum(1 for word in words if word in english_indicators)
                
                # If majority of words are English, it's English
                if len(words) > 0 and english_word_count / len(words) >= 0.5:
                    return "en"
                
                # If text is purely English-looking (no Indian keywords), default to English
                if hindi_matches == 0 and tamil_match_count == 0 and len(words) > 2:
                    return "en"
                
                # For short phrases, be conservative - default to English
                if len(words) <= 3 and hindi_matches == 0 and tamil_match_count == 0:
                    return "en"
            
            # Step 4: Fallback - assume English for undetected Latin script
            return "en"
            
        except Exception as e:
            print(f"[WARN] Language detection failed: {e}")
            return "en"
    
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        if not self.available or source_lang == target_lang:
            return text
        try:
            translator = self.GoogleTranslator(source=source_lang, target=target_lang)
            translated = translator.translate(text)
            
            # Post-process common mistranslations (CRITICAL FIX)
            if source_lang == 'hi' and target_lang == 'en':
                # Fix common Hindi→English translation errors
                text_lower = text.lower()
                translated_lower = translated.lower()
                
                # Sleep-related fixes
                if any(word in text_lower for word in ['neend', 'nind', 'nahi aa rahi', 'nahi aa raha']):
                    if 'sleep' not in translated_lower:
                        translated = "I can't sleep"
                        print(f"[FIX] Corrected sleep translation")
                
                # Hunger fixes
                if any(word in text_lower for word in ['bhook', 'bhookh', 'lagi', 'laga']):
                    if 'hungry' not in translated_lower and 'hunger' not in translated_lower:
                        translated = "I am hungry"
                        print(f"[FIX] Corrected hunger translation")
                
                # Medicine fixes
                if 'dawa' in text_lower or 'dawai' in text_lower or 'dowaai' in text_lower:
                    if 'medicine' not in translated_lower and 'medication' not in translated_lower:
                        if 'lena' in text_lower or 'leni' in text_lower:
                            translated = "I need to take medicine"
                        else:
                            translated = translated.replace('drug', 'medicine').replace('drugs', 'medicine')
                        print(f"[FIX] Corrected medicine translation")
                
                # Pain fixes
                if 'dard' in text_lower:
                    if 'pain' not in translated_lower:
                        translated = translated + " (pain)"
                        print(f"[FIX] Added pain context")
            
            # Tamil→English fixes
            elif source_lang == 'ta' and target_lang == 'en':
                text_lower = text.lower()
                
                # Common Tamil mistranslations
                if 'vali' in text_lower or 'valikuthu' in text_lower:
                    if 'pain' not in translated.lower():
                        translated = "I have pain"
                        print(f"[FIX] Corrected Tamil pain translation")
                
                if 'tired' in text_lower or 'thakku' in text_lower:
                    translated = translated.replace('very very', 'very')  # Common Tamil duplication
            
            # If translation failed (returns same text), try auto-detect
            if translated == text and source_lang != 'auto':
                translator_auto = self.GoogleTranslator(source='auto', target=target_lang)
                translated = translator_auto.translate(text)
            
            return translated
            
        except Exception as e:
            print(f"[WARN] Translation failed: {e}")
            return text

# ============================================================
# Unified Translator Class (Manages All Backends)
# ============================================================
import re
from typing import Tuple

class UnifiedTranslator:
    """
    Unified translator focused on Indian languages + English
    """

    def __init__(self, preferred_backend="google"):
        """
        Args:
            preferred_backend: "google" (only option currently)
        """
        self.backends = {}
        self.preferred = preferred_backend

        # Initialize translation backend
        print("\n" + "=" * 60)
        print("Initializing Translation for Indian Languages + English")
        print("=" * 60)

        # Initialize Google Translator
        google = GoogleTranslatorBackend()
        if google.available:
            self.backends["google"] = google

        if not self.backends:
            print("⚠️ No translation backends available! Install: pip install deep-translator")
        else:
            print("\n✅ Translation ready for Indian languages")
            print("=" * 60 + "\n")

        # Set active backend
        self.active_backend = self.backends.get(preferred_backend)

        # Language name mapping
        self.language_names = {
            "en": "English",
            "hi": "Hindi",
            "ta": "Tamil",
            "te": "Telugu",
            "bn": "Bengali",
            "mr": "Marathi",
            "gu": "Gujarati",
            "kn": "Kannada",
            "ml": "Malayalam",
            "pa": "Punjabi",
            "or": "Odia",
            "as": "Assamese",
            "ur": "Urdu"
        }

    def is_available(self) -> bool:
        """Check if translation is available"""
        return self.active_backend is not None

    # -------------------------------------------------------------------------
    # 💡 UPDATED LANGUAGE DETECTION LOGIC
    # -------------------------------------------------------------------------
    def detect_language(self, text: str) -> Tuple[str, str]:
        """
        Detect language of input text (Indian languages + English only)
        Returns: (language_code, language_name)
        """
        if not self.active_backend:
            return "en", "English"

        try:
            text_lower = text.lower().strip()

            # Step 0: Common English phrases (fast check)
            common_english_phrases = [
                "what's", "what is", "how are", "in the news", "tell me",
                "call my", "call me", "text me", "message my",
                "my son", "my daughter", "my friend", "please help",
                "thank you", "how do", "where is", "when will"
            ]
            if any(p in text_lower for p in common_english_phrases):
                return "en", "English"

            # Step 1: Unicode script detection
            if any('\u0900' <= c <= '\u097F' for c in text):
                return "hi", "Hindi"  # Devanagari
            if any('\u0B80' <= c <= '\u0BFF' for c in text):
                return "ta", "Tamil"
            if any('\u0C00' <= c <= '\u0C7F' for c in text):
                return "te", "Telugu"
            if any('\u0980' <= c <= '\u09FF' for c in text):
                return "bn", "Bengali"
            if any('\u0A80' <= c <= '\u0AFF' for c in text):
                return "gu", "Gujarati"
            if any('\u0C80' <= c <= '\u0CFF' for c in text):
                return "kn", "Kannada"
            if any('\u0D00' <= c <= '\u0D7F' for c in text):
                return "ml", "Malayalam"
            if any('\u0A00' <= c <= '\u0A7F' for c in text):
                return "pa", "Punjabi"

            # Step 2: Romanized (ASCII) text
            is_ascii = all(ord(ch) < 128 or ch.isspace() for ch in text)
            if is_ascii:
                words = re.findall(r"\b[a-z']+\b", text_lower)

                # Strong English signals
                english_common = {
                    "i", "am", "is", "are", "the", "a", "an", "in", "on", "at", "for",
                    "you", "your", "my", "me", "what", "where", "when", "why", "how",
                    "this", "that", "it", "he", "she", "we", "they", "today", "news",
                    "do", "does", "did", "can", "could", "would", "should", "will",
                    "want", "need", "feel", "help", "please", "thank", "good", "bad"
                }
                english_hits = sum(1 for w in words if w in english_common)
                ratio = english_hits / max(1, len(words))

                # ✅ Priority English detection
                if ratio >= 0.4 or english_hits >= 2:
                    return "en", "English"

                # Tamil keywords (romanized)
                tamil_keywords = [
                    "naan", "naanu", "enakku", "enaaku", "vanakkam", "irukku",
                    "pannu", "vena", "vendum", "nalla", "romba", "sapadu", "thanni"
                ]
                if any(re.search(r'\b' + re.escape(w) + r'\b', text_lower) for w in tamil_keywords):
                    return "ta", "Tamil"

                # Hindi keywords (romanized)
                hindi_keywords = [
                    "mujhe", "mujhko", "hai", "hoon", "hain", "tha", "thi", "the",
                    "mera", "meri", "mere", "tumhara", "tumhari", "tumhare",
                    "kya", "kaise", "kyun", "nahi", "nahin", "mat",
                    "aap", "tum", "main", "hum", "madad", "dawa", "so", "khana",
                    "pyaas", "gaya", "raha", "bahut", "thoda", "kuch", "yeh", "woh",
                    "thik", "theek", "achha", "acha", "accha", "bura"
                ]
                if any(re.search(r'\b' + re.escape(w) + r'\b', text_lower) for w in hindi_keywords):
                    return "hi", "Hindi"

                # Telugu keywords
                telugu_keywords = ["nenu", "naaku", "ela", "emiti", "undhi", "chaala"]
                if any(re.search(r'\b' + re.escape(w) + r'\b', text_lower) for w in telugu_keywords):
                    return "te", "Telugu"

                # Bengali keywords
                bengali_keywords = ["ami", "amar", "kemon", "keno", "tomar"]
                if any(re.search(r'\b' + re.escape(w) + r'\b', text_lower) for w in bengali_keywords):
                    return "bn", "Bengali"

                # Default
                return "en", "English"

            # Fallback Latin script → English
            return "en", "English"

        except Exception as e:
            print(f"[WARN] Language detection failed: {e}")
            return "en", "English"

    # -------------------------------------------------------------------------
    # TRANSLATION WRAPPERS
    # -------------------------------------------------------------------------
    def translate(self, text: str, source_lang: str = None, target_lang: str = "en") -> str:
        """Translate text from source to target language"""
        if not self.active_backend:
            return text
        if source_lang is None:
            source_lang, _ = self.detect_language(text)
        if source_lang == target_lang:
            return text
        try:
            return self.active_backend.translate(text, source_lang, target_lang)
        except Exception as e:
            print(f"[WARN] Translation failed: {e}")
            return text

    def translate_to_english(self, text: str) -> Tuple[str, str]:
        """Translate any language to English"""
        lang_code, _ = self.detect_language(text)
        if lang_code == "en":
            return text, lang_code
        translated = self.translate(text, source_lang=lang_code, target_lang="en")
        return translated, lang_code

    def translate_from_english(self, text: str, target_lang: str) -> str:
        """Translate English text to target language"""
        if target_lang == "en":
            return text
        return self.translate(text, source_lang="en", target_lang=target_lang)

    def get_supported_languages(self) -> dict:
        return self.language_names

    def get_active_backend(self) -> str:
        if not self.active_backend:
            return "None"
        return self.active_backend.__class__.__name__


# ============================================================
# Convenience Functions
# ============================================================

# Global translator instance
_translator = None

def get_translator(preferred_backend="google") -> UnifiedTranslator:
    """Get or create global translator instance"""
    global _translator
    if _translator is None:
        _translator = UnifiedTranslator(preferred_backend=preferred_backend)
    return _translator


def detect_language(text: str) -> Tuple[str, str]:
    """Detect language (convenience function)"""
    translator = get_translator()
    return translator.detect_language(text)


def translate_text(text: str, source_lang: str = None, target_lang: str = "en") -> str:
    """Translate text (convenience function)"""
    translator = get_translator()
    return translator.translate(text, source_lang, target_lang)


# ============================================================
# Testing
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing Translation Module - Indian Languages + English")
    print("="*60 + "\n")
    
    # Initialize translator
    translator = UnifiedTranslator(preferred_backend="google")
    
    if not translator.is_available():
        print("❌ No translation backend available!")
        print("Install: pip install deep-translator")
        exit(1)
    
    print(f"Active backend: {translator.get_active_backend()}\n")
    
    # Test cases - Indian languages + English
    test_cases = [
        ("Hello, how are you?", "Should detect English"),
        ("I feel lonely", "Should detect English"),
        ("I am in pain", "Should detect English"),
        ("mujhe pain ho raha hai", "Should detect Romanized Hindi"),
        ("मुझे दर्द हो रहा है", "Should detect Hindi (Devanagari)"),
        ("main bahut thak gaya hoon", "Should detect Romanized Hindi"),
        ("enaaku vali irukku", "Should detect Romanized Tamil - I have pain"),
        ("naan romba tired", "Should detect Romanized Tamil - I am very tired"),
        ("en paiyanukku call pannanum", "Should detect Romanized Tamil - I want to call my son"),
        ("enakku help venum", "Should detect Romanized Tamil - I need help"),
        ("நான் சோகமாக இருக்கிறேன்", "Should detect Tamil (Tamil script)"),
        ("nenu chaala baadhaga unnanu", "Should detect Romanized Telugu"),
        ("నాకు నొప్పిగా ఉంది", "Should detect Telugu (Telugu script)"),
        ("আমি একা অনুভব করছি", "Should detect Bengali"),
    ]
    
    print("Testing language detection and translation:\n")
    for text, expected in test_cases:
        lang_code, lang_name = translator.detect_language(text)
        print(f"Input: {text}")
        print(f"Expected: {expected}")
        print(f"Detected: {lang_name} ({lang_code})")
        
        if lang_code != "en":
            translated, _ = translator.translate_to_english(text)
            print(f"→ English: {translated}")
            
            # Translate back
            back_translated = translator.translate_from_english(translated, lang_code)
            print(f"→ Back to {lang_name}: {back_translated}")
        
        print("-" * 60)
    
    print("\n✅ Translation module test complete!")
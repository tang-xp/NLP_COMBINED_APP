# smart_text_cleaner.py

import re
import spacy
from spellchecker import SpellChecker
from typing import List, Tuple, Dict, Optional
import Levenshtein  # pip install python-Levenshtein
from collections import defaultdict
import pickle
import os

class SmartSpellChecker:
    """
    Context-aware spell checker that learns and adapts rather than using hardcoded rules.
    """
    
    def __init__(self, model_path: str = "smart_spell_model.pkl"):
        self.model_path = model_path
        self.spell = SpellChecker()
        
        # Initialize spaCy for context analysis
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model not found. Context features will be limited.")
            self.nlp = None
        
        # Learning components
        self.word_patterns = defaultdict(list)
        self.context_corrections = defaultdict(dict)
        self.confidence_scores = defaultdict(float)
        self.learned_vocabulary = set()
        
        # Load existing model if available
        self.load_model()
        
    def get_word_context(self, text: str, word_pos: int) -> Dict:
        """Extract contextual features around a word."""
        words = text.split()
        context = {
            'prev_word': words[word_pos-1].lower() if word_pos > 0 else None,
            'next_word': words[word_pos+1].lower() if word_pos < len(words)-1 else None,
            'sentence_length': len(words),
            'word_position': word_pos / len(words),
        }
        if self.nlp:
            try:
                doc = self.nlp(text)
                tokens = list(doc)
                if word_pos < len(tokens):
                    token = tokens[word_pos]
                    context.update({
                        'pos_tag': token.pos_,
                        'is_entity': token.ent_type_ != '',
                        'entity_type': token.ent_type_,
                        'dependency': token.dep_,
                    })
            except: pass
        return context
    
    def is_likely_proper_noun_smart(self, word: str, context: Dict) -> bool:
        """Smart proper noun detection using multiple signals."""
        clean_word = re.sub(r'[^\w]', '', word)
        if not clean_word or not clean_word[0].isupper(): return False
        if context.get('is_entity', False): return context.get('entity_type') in ['PERSON', 'ORG', 'GPE']
        prev_word = context.get('prev_word', '')
        if prev_word in ['mr', 'mrs', 'ms', 'dr', 'prof']: return True
        if context.get('word_position', 0) < 0.2: return True
        return False

    # --- THIS IS THE CORRECTED get_correction_candidates METHOD ---
    def get_correction_candidates(self, word: str) -> List[Tuple[str, float]]:
        """Get correction candidates with confidence scores."""
        candidates = []
        word_lower = word.lower()

        # Get the set of candidates from the spell checker
        spell_candidates = self.spell.candidates(word_lower)
        
        # If the spell checker returns nothing, we have no candidates to suggest
        if spell_candidates is None:
            return []

        for candidate in list(spell_candidates)[:5]:  # Top 5 suggestions
            # Calculate a confidence score based on word similarity (Levenshtein distance)
            distance = Levenshtein.distance(word_lower, candidate)
            normalized_distance = distance / max(len(word_lower), len(candidate), 1)
            confidence = 1.0 - normalized_distance

            # Boost confidence for candidates that are in our custom learned vocabulary (less likely)
            if candidate in self.learned_vocabulary:
                confidence = min(1.0, confidence + 0.1) # small boost
            
            candidates.append((candidate, confidence))
        
        # Check learned patterns (if any)
        for pattern_word, corrections in self.word_patterns.items():
            if self.is_similar_pattern(word_lower, pattern_word):
                for correction, freq in corrections:
                    candidates.append((correction, freq / 10.0))

        # Remove duplicates, keeping the one with the highest confidence
        unique_candidates = {}
        for candidate, confidence in candidates:
            if candidate not in unique_candidates or confidence > unique_candidates[candidate]:
                unique_candidates[candidate] = confidence
        
        # Sort the unique candidates by confidence
        sorted_candidates = sorted(unique_candidates.items(), key=lambda item: item[1], reverse=True)
        
        return sorted_candidates[:3]  # Return the top 3 best candidates

    def is_similar_pattern(self, word1: str, word2: str) -> bool:
        """Check if two words have similar character patterns."""
        if abs(len(word1) - len(word2)) > 2: return False
        return Levenshtein.distance(word1, word2) <= 2

    # --- THIS IS THE NEW is_ordinal_number METHOD, PLACED CORRECTLY INSIDE THE CLASS ---
    def is_ordinal_number(self, word: str) -> bool:
        """Check if a word is an ordinal number like 1st, 2nd, 15th."""
        return bool(re.match(r'^\d+(st|nd|rd|th)$', word.lower()))

    # --- THIS IS THE CORRECTED should_correct_word METHOD ---
    def should_correct_word(self, word: str, context: Dict, candidates: List[Tuple[str, float]]) -> bool:
        """Intelligent decision on whether to correct a word."""
        if not candidates: return False
        
        best_candidate, confidence = candidates[0]
        
        if confidence < 0.6: return False # Increased confidence threshold
        if self.is_likely_proper_noun_smart(word, context) and confidence < 0.8: return False
        if word.lower() in self.learned_vocabulary: return False
        if re.match(r'^\d{1,2}(:\d{2})?(am|pm)$', word.lower()): return False
        if self.is_ordinal_number(word): return False # Correctly calls the method
        
        return True
    
    def learn_from_correction(self, original: str, corrected: str, context: Dict):
        """Learn from successful corrections."""
        if original.lower() != corrected.lower():
            self.word_patterns[original.lower()].append((corrected, 1))
            context_key = f"{context.get('prev_word', '')}_{context.get('pos_tag', '')}"
            if context_key not in self.context_corrections:
                self.context_corrections[context_key] = {}
            self.context_corrections[context_key][original.lower()] = corrected
            self.confidence_scores[f"{original.lower()}_{corrected.lower()}"] += 0.1
    
    def add_to_vocabulary(self, words: List[str]):
        """Add words to learned vocabulary (words that should not be corrected)."""
        for word in words:
            self.learned_vocabulary.add(word.lower())
    
    def correct_text_smart(self, text: str, learn: bool = True, verbose: bool = False) -> str:
        """Smart text correction with learning capabilities."""
        if not text or not text.strip(): return text
        words = text.split()
        corrected_words = []
        corrections_made = []
        for i, word in enumerate(words):
            clean_word, prefix, suffix = self._clean_punctuation(word)
            if not clean_word:
                corrected_words.append(word)
                continue
            context = self.get_word_context(text, i)
            if clean_word.lower() not in self.spell:
                candidates = self.get_correction_candidates(clean_word)
                if self.should_correct_word(clean_word, context, candidates):
                    best_candidate, confidence = candidates[0]
                    if clean_word.isupper(): corrected_clean = best_candidate.upper()
                    elif clean_word[0].isupper(): corrected_clean = best_candidate.capitalize()
                    else: corrected_clean = best_candidate
                    corrected_word = prefix + corrected_clean + suffix
                    corrected_words.append(corrected_word)
                    corrections_made.append((word, corrected_word, confidence))
                    if learn: self.learn_from_correction(clean_word, corrected_clean, context)
                else: corrected_words.append(word)
            else: corrected_words.append(word)
        result = " ".join(corrected_words)
        if verbose and corrections_made:
            print(f"\n--- Smart Spell Correction Applied ---")
            for original, corrected, conf in corrections_made:
                print(f"  -> {original} → {corrected} (confidence: {conf:.2f})")
            print(f"  -> Final text: '{result}'")
        return result
    
    def _clean_punctuation(self, word: str) -> Tuple[str, str, str]:
        """Helper method to clean punctuation."""
        prefix_match = re.match(r'^([^\w]*)', word)
        prefix = prefix_match.group(1) if prefix_match else ''
        suffix_match = re.search(r'([^\w]*)$', word)
        suffix = suffix_match.group(1) if suffix_match else ''
        clean_word = word[len(prefix):len(word)-len(suffix)] if suffix else word[len(prefix):]
        return clean_word, prefix, suffix
    
    def save_model(self):
        """Save learned patterns and vocabulary."""
        model_data = { 'word_patterns': dict(self.word_patterns), 'context_corrections': dict(self.context_corrections), 'confidence_scores': dict(self.confidence_scores), 'learned_vocabulary': self.learned_vocabulary }
        try:
            with open(self.model_path, 'wb') as f: pickle.dump(model_data, f)
            print(f"Model saved to {self.model_path}")
        except Exception as e: print(f"Error saving model: {e}")
    
    def load_model(self):
        """Load previously learned patterns."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f: model_data = pickle.load(f)
                self.word_patterns = defaultdict(list, model_data.get('word_patterns', {}))
                self.context_corrections = defaultdict(dict, model_data.get('context_corrections', {}))
                self.confidence_scores = defaultdict(float, model_data.get('confidence_scores', {}))
                self.learned_vocabulary = model_data.get('learned_vocabulary', set())
                print(f"Loaded model from {self.model_path}")
            except Exception as e: print(f"Error loading model: {e}")

# Factory function for backward compatibility
def create_smart_spell_checker() -> SmartSpellChecker:
    """Create and configure a smart spell checker."""
    checker = SmartSpellChecker()
    initial_vocab = ['jira', 'github', 'slack', 'zoom', 'teams', 'api', 'json', 'xml', 'sql']
    checker.add_to_vocabulary(initial_vocab)
    return checker

# Backward compatibility function
def correct_text(text: str) -> str:
    """Backward compatible interface."""
    global _global_checker
    if '_global_checker' not in globals():
        _global_checker = create_smart_spell_checker()
    return _global_checker.correct_text_smart(text, learn=True, verbose=True)

# Example usage and testing
if __name__ == "__main__":
    print("=" * 70)
    print("    SMART CONTEXT-AWARE SPELL CHECKER TEST")
    print("=" * 70)
    checker = create_smart_spell_checker()
    test_cases = [
        "Let's have a meting with John at 9pm tomorow.",
        "My appointent is on Fryday.",
        "The conferance is sceduled for Febuary.",
        "Septmber is a beutiful month.",
        "Dr. Smith will attnd the workshp.",
        "The GitHub API rturns JSON data.",
        "Ocotber 15th is the deadlne.",
        "Alex schedled a cal for Thursdy.",
        "Let's meat at the restaurant.",
        "I ate too much bred.",
    ]
    print("\n🧠 Testing Smart Spell Checker...")
    for i, text in enumerate(test_cases, 1):
        print(f"\n--- Test {i} ---")
        print(f"Original:  '{text}'")
        corrected = checker.correct_text_smart(text, learn=True, verbose=True)
        print(f"Corrected: '{corrected}'")
    print(f"\n🎓 Learning Test:")
    print("Adding 'Jira' and 'GitHub' to vocabulary...")
    checker.add_to_vocabulary(['Jira', 'GitHub'])
    test_text = "The Jra ticket is linkd to GitHb."
    print(f"Before learning: '{test_text}'")
    corrected = checker.correct_text_smart(test_text, learn=True, verbose=True)
    print(f"After correction: '{corrected}'")
    checker.save_model()
    print("\n" + "=" * 70)
    print("    SMART SPELL CHECKER TEST COMPLETE")
    print("=" * 70)
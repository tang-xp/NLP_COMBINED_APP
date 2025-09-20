import os
import re
import pickle
import spacy
from collections import defaultdict
from spellchecker import SpellChecker
from typing import List, Tuple, Dict
import Levenshtein  # pip install python-Levenshtein


class SmartSpellChecker:
    """
    Context-aware spell checker that learns and adapts.
    """

    def __init__(self, model_path: str = "smart_spell_model.pkl"):
        self.model_path = model_path
        self.spell = SpellChecker()

        # Load spaCy model if available
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model not found. Context features limited.")
            self.nlp = None

        # Learning components
        self.word_patterns = defaultdict(list)
        self.context_corrections = defaultdict(dict)
        self.confidence_scores = defaultdict(float)
        self.learned_vocabulary = set()

        self.load_model()

    # ---------- Context helpers ----------
    def get_word_context(self, text: str, word_pos: int) -> Dict:
        words = text.split()
        context = {
            "prev_word": words[word_pos - 1].lower() if word_pos > 0 else None,
            "next_word": words[word_pos + 1].lower() if word_pos < len(words) - 1 else None,
            "sentence_length": len(words),
            "word_position": word_pos / len(words),
        }
        if self.nlp:
            try:
                doc = self.nlp(text)
                tokens = list(doc)
                if word_pos < len(tokens):
                    token = tokens[word_pos]
                    context.update({
                        "pos_tag": token.pos_,
                        "is_entity": token.ent_type_ != "",
                        "entity_type": token.ent_type_,
                        "dependency": token.dep_,
                    })
            except:
                pass
        return context

    def is_likely_proper_noun_smart(self, word: str, context: Dict) -> bool:
        clean_word = re.sub(r"[^\w]", "", word)
        if not clean_word or not clean_word[0].isupper():
            return False
        if context.get("is_entity", False):
            return context.get("entity_type") in ["PERSON", "ORG", "GPE"]
        if context.get("prev_word", "") in ["mr", "mrs", "ms", "dr", "prof"]:
            return True
        return context.get("word_position", 0) < 0.2

    def is_ordinal_number(self, word: str) -> bool:
        return bool(re.match(r"^\d+(st|nd|rd|th)$", word.lower()))

    # ---------- Candidate generation ----------
    def get_correction_candidates(self, word: str) -> List[Tuple[str, float]]:
        candidates = []
        w = word.lower()
        spell_candidates = self.spell.candidates(w)
        if not spell_candidates:
            return []

        for c in list(spell_candidates)[:5]:
            distance = Levenshtein.distance(w, c)
            confidence = 1.0 - distance / max(len(w), len(c), 1)
            if c in self.learned_vocabulary:
                confidence = min(1.0, confidence + 0.1)
            candidates.append((c, confidence))

        for pattern_word, corrections in self.word_patterns.items():
            if self.is_similar_pattern(w, pattern_word):
                for corr, freq in corrections:
                    candidates.append((corr, freq / 10.0))

        # keep highest-confidence unique candidates
        uniq = {}
        for c, conf in candidates:
            if c not in uniq or conf > uniq[c]:
                uniq[c] = conf
        return sorted(uniq.items(), key=lambda x: x[1], reverse=True)[:3]

    def is_similar_pattern(self, w1: str, w2: str) -> bool:
        return abs(len(w1) - len(w2)) <= 2 and Levenshtein.distance(w1, w2) <= 2

    def should_correct_word(self, word: str, context: Dict,
                            candidates: List[Tuple[str, float]]) -> bool:
        if not candidates:
            return False
        best, conf = candidates[0]
        if conf < 0.6:
            return False
        if self.is_likely_proper_noun_smart(word, context) and conf < 0.8:
            return False
        if word.lower() in self.learned_vocabulary:
            return False
        if re.match(r"^\d{1,2}(:\d{2})?(am|pm)$", word.lower()):
            return False
        if self.is_ordinal_number(word):
            return False
        return True

    # ---------- Learning ----------
    def learn_from_correction(self, original: str, corrected: str, context: Dict):
        if original.lower() != corrected.lower():
            self.word_patterns[original.lower()].append((corrected, 1))
            key = f"{context.get('prev_word','')}_{context.get('pos_tag','')}"
            self.context_corrections.setdefault(key, {})[original.lower()] = corrected
            self.confidence_scores[f"{original.lower()}_{corrected.lower()}"] += 0.1

    def add_to_vocabulary(self, words: List[str]):
        for w in words:
            self.learned_vocabulary.add(w.lower())

    # ---------- Main correction ----------
    def correct_text_smart(self, text: str,
                           learn: bool = True,
                           verbose: bool = False) -> str:
        if not text or not text.strip():
            return text
        words = text.split()
        corrected_words, corrections_made = [], []
        for i, word in enumerate(words):
            clean, pre, suf = self._clean_punctuation(word)
            if not clean:
                corrected_words.append(word)
                continue
            context = self.get_word_context(text, i)
            if clean.lower() not in self.spell:
                candidates = self.get_correction_candidates(clean)
                if self.should_correct_word(clean, context, candidates):
                    best, conf = candidates[0]
                    if clean.isupper():
                        fixed = best.upper()
                    elif clean[0].isupper():
                        fixed = best.capitalize()
                    else:
                        fixed = best
                    corrected_words.append(pre + fixed + suf)
                    corrections_made.append((word, pre + fixed + suf, conf))
                    if learn:
                        self.learn_from_correction(clean, fixed, context)
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)
        result = " ".join(corrected_words)
        if verbose and corrections_made:
            print("\n--- Smart Spell Correction ---")
            for o, c, conf in corrections_made:
                print(f"  -> {o} → {c} (conf: {conf:.2f})")
            print(f"  -> Final: {result}")
        return result

    def _clean_punctuation(self, word: str) -> Tuple[str, str, str]:
        pre = re.match(r"^([^\w]*)", word).group(1) or ""
        suf = re.search(r"([^\w]*)$", word).group(1) or ""
        core = word[len(pre):len(word) - len(suf)] if suf else word[len(pre):]
        return core, pre, suf

    # ---------- Persistence ----------
    def save_model(self):
        data = {
            "word_patterns": dict(self.word_patterns),
            "context_corrections": dict(self.context_corrections),
            "confidence_scores": dict(self.confidence_scores),
            "learned_vocabulary": self.learned_vocabulary,
        }
        with open(self.model_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                d = pickle.load(f)
            self.word_patterns = defaultdict(list, d.get("word_patterns", {}))
            self.context_corrections = defaultdict(dict, d.get("context_corrections", {}))
            self.confidence_scores = defaultdict(float, d.get("confidence_scores", {}))
            self.learned_vocabulary = d.get("learned_vocabulary", set())
            print(f"Loaded model from {self.model_path}")


def create_smart_spell_checker() -> SmartSpellChecker:
    checker = SmartSpellChecker()
    checker.add_to_vocabulary(
        ["jira", "github", "slack", "zoom", "teams", "api", "json", "xml", "sql"]
    )
    return checker


def correct_text(text: str) -> str:
    global _global_checker
    if "_global_checker" not in globals():
        _global_checker = create_smart_spell_checker()
    return _global_checker.correct_text_smart(text, learn=True, verbose=True)


if __name__ == "__main__":
    checker = create_smart_spell_checker()
    sample = "Let's have a meting with John at 9pm tomorow."
    print(correct_text(sample))

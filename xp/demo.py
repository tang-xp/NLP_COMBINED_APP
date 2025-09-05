import spacy
import joblib
from datetime import datetime, timedelta
from temporal_normalizer import normalize_temporal_expression
import re

class FeatureExtractor:
    def __init__(self):
        # Enhanced lexicons with more comprehensive coverage
        self.time_indicators = {
            "am", "pm", "morning", "afternoon", "evening", "night", "noon", "midnight",
            "dawn", "dusk", "early", "late", "sharp", "exactly", "around", "about"
        }
        
        self.date_words = {
            # Days
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
            "mon", "tue", "wed", "thu", "fri", "sat", "sun",
            # Months
            "january", "february", "march", "april", "may", "june", "july", "august", 
            "september", "october", "november", "december",
            "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
            # Relative dates
            "today", "tomorrow", "yesterday", "week", "month", "year", "day",
            "next", "last", "this", "coming", "past", "future"
        }
        
        self.event_words = {
            "meeting", "conference", "appointment", "deadline", "review", "lunch", "dinner",
            "breakfast", "session", "workshop", "seminar", "call", "interview", "presentation",
            "training", "briefing", "attack", "audit", "flight", "debrief", "sync", "huddle",
            "standup", "demo", "retrospective", "planning", "ceremony", "event", "party",
            "celebration", "gathering", "checkup", "consultation", "visit"
        }
        
        # More flexible person indicators instead of specific names
        self.person_indicators = {
            "mr", "mrs", "ms", "dr", "prof", "professor", "sir", "madam", "miss"
        }
        
        self.prepositions = {
            "by", "at", "on", "in", "for", "during", "before", "after", "to", "from",
            "with", "until", "since", "between", "among", "through", "throughout"
        }
        
        # Compiled regex patterns for better performance
        self.time_patterns = [
            re.compile(r"\d{1,2}:\d{2}(?::\d{2})?(?:am|pm)?", re.IGNORECASE),  # 12:30, 12:30pm
            re.compile(r"\d{1,2}(?:am|pm)", re.IGNORECASE),  # 12pm
            re.compile(r"\d{1,2}h\d{2}", re.IGNORECASE),  # 14h30
        ]
        
        self.date_patterns = [
            re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}"),  # 12/25/2023
            re.compile(r"\d{1,2}-\d{1,2}-\d{2,4}"),  # 12-25-2023
            re.compile(r"\d{4}-\d{1,2}-\d{1,2}"),    # 2023-12-25
            re.compile(r"\d{1,2}(?:st|nd|rd|th)", re.IGNORECASE),  # 1st, 2nd, 3rd
        ]

    def get_word_shape(self, word):
        """Generate word shape feature"""
        shape = ""
        for char in word:
            if char.isupper():
                shape += "X"
            elif char.islower():
                shape += "x"
            elif char.isdigit():
                shape += "d"
            else:
                shape += char
        return shape

    def matches_time_pattern(self, word):
        """Check if word matches any time pattern"""
        return any(pattern.match(word) for pattern in self.time_patterns)

    def matches_date_pattern(self, word):
        """Check if word matches any date pattern"""
        return any(pattern.match(word) for pattern in self.date_patterns)

    def enhanced_word_features(self, sent, i):
        """Enhanced feature engineering with better error handling"""
        word, postag = sent[i]
        word_lower = word.lower()
        
        # Base features
        features = {
            "bias": 1.0,
            "word.lower()": word_lower,
            "word.isdigit()": word.isdigit(),
            "word.istitle()": word.istitle(),
            "word.isupper()": word.isupper(),
            "word.length": len(word),
            "postag": postag,
        }
        
        # Lexicon features
        features.update({
            'is_time_word': word_lower in self.time_indicators,
            'is_date_word': word_lower in self.date_words,
            'is_event_word': word_lower in self.event_words,
            'is_person_indicator': word_lower in self.person_indicators,
            'is_preposition': word_lower in self.prepositions,
            'matches_time_pattern': self.matches_time_pattern(word),
            'matches_date_pattern': self.matches_date_pattern(word),
        })
        
        # Word shape and character features
        features["word_shape"] = self.get_word_shape(word)
        features["has_digit"] = any(c.isdigit() for c in word)
        features["has_punct"] = any(c in ".,!?;:" for c in word)
        features["starts_with_cap"] = word[0].isupper() if word else False
        
        # Contextual features (previous word)
        if i > 0:
            prev_word, prev_postag = sent[i - 1]
            prev_word_lower = prev_word.lower()
            features.update({
                "prev_word.lower()": prev_word_lower,
                "prev_postag": prev_postag,
                "prev_is_preposition": prev_word_lower in self.prepositions,
                "prev_is_time": prev_word_lower in self.time_indicators,
                "prev_word_shape": self.get_word_shape(prev_word),
            })
            # Bigram feature
            features["prev_current_bigram"] = f"{prev_word_lower}_{word_lower}"
        else:
            features["BOS"] = True
        
        # Contextual features (next word)
        if i < len(sent) - 1:
            next_word, next_postag = sent[i + 1]
            next_word_lower = next_word.lower()
            features.update({
                "next_word.lower()": next_word_lower,
                "next_postag": next_postag,
                "next_is_time_word": next_word_lower in self.time_indicators,
                "next_is_date_word": next_word_lower in self.date_words,
                "next_word_shape": self.get_word_shape(next_word),
            })
            # Bigram feature
            features["current_next_bigram"] = f"{word_lower}_{next_word_lower}"
        else:
            features["EOS"] = True
        
        # Trigram features (if available)
        if 0 < i < len(sent) - 1:
            prev_word, _ = sent[i - 1]
            next_word, _ = sent[i + 1]
            features["trigram"] = f"{prev_word.lower()}_{word_lower}_{next_word.lower()}"
        
        return features

# --- 1. Load Trained Models & SpaCy for Tokenization ---
print("Loading spaCy model for tokenization and POS tagging...")
nlp = spacy.load("en_core_web_sm")
print("Loading trained CRF model (crf_final_model.joblib)...")



try:
    model_data = joblib.load("xp/xp_crf_model.joblib")
    
    # Handle both old format (just model) and new format (dict with model + metadata)
    if isinstance(model_data, dict):
        crf_model = model_data['model']
        feature_extractor = model_data.get('feature_extractor', None)
        print("  -> CRF model with metadata loaded successfully.")
    else:
        # This handles the old format where only the model was saved.
        crf_model = model_data
        feature_extractor = None
        print("  -> CRF model (legacy format) loaded successfully.")
        
except FileNotFoundError:
    print("❌ ERROR: 'xp/xp_crf_model.joblib' not found. Please run the train_crf.py script first.")
    exit()

# --- 2. Feature Engineering Class (Updated to match training) ---
class FeatureExtractor:
    def __init__(self):
        # Enhanced lexicons with more comprehensive coverage
        self.time_indicators = {
            "am", "pm", "morning", "afternoon", "evening", "night", "noon", "midnight",
            "dawn", "dusk", "early", "late", "sharp", "exactly", "around", "about"
        }
        
        self.date_words = {
            # Days
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
            "mon", "tue", "wed", "thu", "fri", "sat", "sun",
            # Months
            "january", "february", "march", "april", "may", "june", "july", "august", 
            "september", "october", "november", "december",
            "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
            # Relative dates
            "today", "tomorrow", "yesterday", "week", "month", "year", "day",
            "next", "last", "this", "coming", "past", "future"
        }
        
        self.event_words = {
            "meeting", "conference", "appointment", "deadline", "review", "lunch", "dinner",
            "breakfast", "session", "workshop", "seminar", "call", "interview", "presentation",
            "training", "briefing", "attack", "audit", "flight", "debrief", "sync", "huddle",
            "standup", "demo", "retrospective", "planning", "ceremony", "event", "party",
            "celebration", "gathering", "checkup", "consultation", "visit"
        }
        
        self.person_indicators = {
            "mr", "mrs", "ms", "dr", "prof", "professor", "sir", "madam", "miss"
        }
        
        self.prepositions = {
            "by", "at", "on", "in", "for", "during", "before", "after", "to", "from",
            "with", "until", "since", "between", "among", "through", "throughout"
        }
        
        # Compiled regex patterns for better performance
        self.time_patterns = [
            re.compile(r"\d{1,2}:\d{2}(?::\d{2})?(?:am|pm)?", re.IGNORECASE),
            re.compile(r"\d{1,2}(?:am|pm)", re.IGNORECASE),
            re.compile(r"\d{1,2}h\d{2}", re.IGNORECASE),
        ]
        
        self.date_patterns = [
            re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}"),
            re.compile(r"\d{1,2}-\d{1,2}-\d{2,4}"),
            re.compile(r"\d{4}-\d{1,2}-\d{1,2}"),
            re.compile(r"\d{1,2}(?:st|nd|rd|th)", re.IGNORECASE),
        ]

    def get_word_shape(self, word):
        """Generate word shape feature"""
        shape = ""
        for char in word:
            if char.isupper():
                shape += "X"
            elif char.islower():
                shape += "x"
            elif char.isdigit():
                shape += "d"
            else:
                shape += char
        return shape

    def matches_time_pattern(self, word):
        """Check if word matches any time pattern"""
        return any(pattern.match(word) for pattern in self.time_patterns)

    def matches_date_pattern(self, word):
        """Check if word matches any date pattern"""
        return any(pattern.match(word) for pattern in self.date_patterns)

    def enhanced_word_features(self, sent, i):
        """Enhanced feature engineering matching training version"""
        word, postag = sent[i]
        word_lower = word.lower()
        
        # Base features
        features = {
            "bias": 1.0,
            "word.lower()": word_lower,
            "word.isdigit()": word.isdigit(),
            "word.istitle()": word.istitle(),
            "word.isupper()": word.isupper(),
            "word.length": len(word),
            "postag": postag,
        }
        
        # Lexicon features
        features.update({
            'is_time_word': word_lower in self.time_indicators,
            'is_date_word': word_lower in self.date_words,
            'is_event_word': word_lower in self.event_words,
            'is_person_indicator': word_lower in self.person_indicators,
            'is_preposition': word_lower in self.prepositions,
            'matches_time_pattern': self.matches_time_pattern(word),
            'matches_date_pattern': self.matches_date_pattern(word),
        })
        
        # Word shape and character features
        features["word_shape"] = self.get_word_shape(word)
        features["has_digit"] = any(c.isdigit() for c in word)
        features["has_punct"] = any(c in ".,!?;:" for c in word)
        features["starts_with_cap"] = word[0].isupper() if word else False
        
        # Contextual features (previous word)
        if i > 0:
            prev_word, prev_postag = sent[i - 1]
            prev_word_lower = prev_word.lower()
            features.update({
                "prev_word.lower()": prev_word_lower,
                "prev_postag": prev_postag,
                "prev_is_preposition": prev_word_lower in self.prepositions,
                "prev_is_time": prev_word_lower in self.time_indicators,
                "prev_word_shape": self.get_word_shape(prev_word),
            })
            features["prev_current_bigram"] = f"{prev_word_lower}_{word_lower}"
        else:
            features["BOS"] = True
        
        # Contextual features (next word)
        if i < len(sent) - 1:
            next_word, next_postag = sent[i + 1]
            next_word_lower = next_word.lower()
            features.update({
                "next_word.lower()": next_word_lower,
                "next_postag": next_postag,
                "next_is_time_word": next_word_lower in self.time_indicators,
                "next_is_date_word": next_word_lower in self.date_words,
                "next_word_shape": self.get_word_shape(next_word),
            })
            features["current_next_bigram"] = f"{word_lower}_{next_word_lower}"
        else:
            features["EOS"] = True
        
        # Trigram features (if available)
        if 0 < i < len(sent) - 1:
            prev_word, _ = sent[i - 1]
            next_word, _ = sent[i + 1]
            features["trigram"] = f"{prev_word.lower()}_{word_lower}_{next_word.lower()}"
        
        return features

# Initialize feature extractor (use loaded one if available, otherwise create new)
if feature_extractor is None:
    feature_extractor = FeatureExtractor()

def sent2features(sent_tokens):
    """Convert sentence tokens to features with improved error handling"""
    if not sent_tokens:
        return []
    
    try:
        text = " ".join(sent_tokens)
        doc = nlp(text)
        pos_tagged_sent = [(token.text, token.pos_) for token in doc]
        
        # Handle tokenization mismatch
        if len(pos_tagged_sent) != len(sent_tokens):
            print(f"Warning: Tokenization mismatch. Expected {len(sent_tokens)}, got {len(pos_tagged_sent)}")
            # Fallback: use original tokens with default POS
            pos_tagged_sent = [(token, "NOUN") for token in sent_tokens]
        
        return [feature_extractor.enhanced_word_features(pos_tagged_sent, i) 
                for i in range(len(pos_tagged_sent))]
    
    except Exception as e:
        print(f"Error in feature extraction: {e}")
        return []

# --- 3. Improved Post-processing BIO to Structured Output ---
def bio_to_structured(tokens, bio_labels):
    """Convert BIO labels to structured output with better handling"""
    structured_output = {"EVENT": [], "DATE": [], "TIME": []}
    current_entity_tokens = []
    current_label_type = None

    for i, (token, label) in enumerate(zip(tokens, bio_labels)):
        if label.startswith("B-"):
            # Save previous entity if exists
            if current_entity_tokens and current_label_type:
                entity_text = " ".join(current_entity_tokens)
                structured_output[current_label_type].append(entity_text)
            
            # Start new entity
            current_entity_tokens = [token]
            current_label_type = label[2:]
            
        elif label.startswith("I-") and current_label_type == label[2:]:
            # Continue current entity
            current_entity_tokens.append(token)
            
        else:  # 'O' or mismatching 'I-' tag
            # Save current entity if exists
            if current_entity_tokens and current_label_type:
                entity_text = " ".join(current_entity_tokens)
                structured_output[current_label_type].append(entity_text)
            
            # Handle orphaned I- tags (treat as B-)
            if label.startswith("I-"):
                current_entity_tokens = [token]
                current_label_type = label[2:]
            else:
                current_entity_tokens = []
                current_label_type = None

    # Save the last entity if loop ends with one
    if current_entity_tokens and current_label_type:
        entity_text = " ".join(current_entity_tokens)
        structured_output[current_label_type].append(entity_text)

    return structured_output

# --- 4. Main Demo Execution ---
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("      TEMPORAL ANALYSIS DEMO (CRF NER + Custom TEN)")
    print("=" * 70)

    # Use actual current date for better demo
    current_date = datetime.now()
    print(f"Reference Date: {current_date.strftime('%Y-%m-%d')} ({current_date.strftime('%A')})")

    test_sentences = [
        "The board meeting is scheduled for tomorrow.",
        "Let's have a meeting with John at 9pm.",
        "The attack happened on Friday.",
        "The conference will be held in January.",
        "Submit the final report by 3:00 PM next Tuesday.",
        "My appointment is at 10 AM on December 25th, 2024.",
        "Lunch is planned for noon today.",
        "Rachel has a presentation on November 15th.",
        "The workshop runs from 9 AM to 5 PM next Monday.",
        "The audit needs to be completed by March 31st.",
        "We are flying to London next week.",
    ]

    for sentence in test_sentences:
        print(f"\n--- Processing Input Text ---\n'{sentence}'")

        # --- Stage 1: NER using your TRAINABLE CRF model ---
        doc = nlp(sentence)
        tokens = [token.text for token in doc]

        features = sent2features(tokens)
        if not features:
            print("-> Could not extract features for NER prediction.")
            print("-> No structured output generated.")
            continue

        predicted_bio_labels = crf_model.predict([features])[0]

        print("\n--- Predicted BIO Labels (from CRF) ---")
        print(f"Tokens: {tokens}")
        print(f"Labels: {predicted_bio_labels}")

        extracted_structured_output = bio_to_structured(tokens, predicted_bio_labels)

        # --- Stage 2: TEN (Normalization) using your custom rule-based logic ---
        final_output = {"EVENT": [], "DATE": [], "TIME": []}

        # Process Events (no normalization beyond extraction)
        final_output["EVENT"] = extracted_structured_output["EVENT"]

        # Process Dates
        for date_text in extracted_structured_output["DATE"]:
            try:
                normalized_date = normalize_temporal_expression(date_text, "DATE", reference_date=current_date)
                final_output["DATE"].append(f"{date_text} ({normalized_date})")
            except Exception as e:
                print(f"Error normalizing date '{date_text}': {e}")
                final_output["DATE"].append(f"{date_text} (normalization failed)")

        # Process Times
        for time_text in extracted_structured_output["TIME"]:
            try:
                normalized_time = normalize_temporal_expression(time_text, "TIME", reference_date=current_date)
                final_output["TIME"].append(f"{time_text} ({normalized_time})")
            except Exception as e:
                print(f"Error normalizing time '{time_text}': {e}")
                final_output["TIME"].append(f"{time_text} (normalization failed)")

        print("\n--- Final Structured Output (NER + TEN) ---")
        for label_type, entities in final_output.items():
            if entities:
                print(f"  {label_type}: {', '.join(entities)}")
            else:
                print(f"  {label_type}: None")
        print("-----------------------------")

    print("\n" + "=" * 70)
    print("Demo complete. Output reflects CRF NER performance")
    print("and custom rule-based Temporal Expression Normalization.")
    print("=" * 70)
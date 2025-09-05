# xp/xp_pipeline.py

import spacy
import joblib
from datetime import datetime
import re
import os
from .xp_post_processor import EnhancedEventExtractor # This should be here
from .temporal_normalizer import normalize_temporal_expression
from .text_cleaner import SmartSpellChecker
from .xp_features import FeatureExtractor

feature_extractor = FeatureExtractor()


script_dir = os.path.dirname(os.path.abspath(__file__))
# --- 1. Load Models and Tools (loaded only once when the app starts) ---
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

# Correctly locate the model file relative to this script
# This assumes xp_crf_model.joblib is in the ROOT folder (NLP_COMBINED_APP)
MODEL_PATH = os.path.join(script_dir, 'xp_crf_model.joblib')
print(f"Loading CRF model from: {MODEL_PATH}")
try:
    crf_model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    print(f"FATAL ERROR: Model not found at {MODEL_PATH}. Make sure 'xp_crf_model.joblib' is in the main project directory.")
    crf_model = None

# Initialize your smart text cleaner
spell_checker = SmartSpellChecker()


def sent2features(sent_tokens):
    """
    Convert sentence tokens to features, with robust handling for empty or mismatched inputs.
    """
    # --- FIX 1: Handle empty input gracefully ---
    if not sent_tokens:
        return [] # Always return a list

    doc = nlp(" ".join(sent_tokens))
    
    # --- FIX 2: Correctly handle tokenization mismatches ---
    # If spaCy's tokenization doesn't match the original, we cannot safely create features.
    # The safest action is to return an empty list, which will cause the sentence to be skipped.
    if len(doc) != len(sent_tokens):
        return [] # Return an empty list to signal a mismatch

    # If everything is fine, proceed with the full feature extraction
    chunk_info = [{'tag': 'O', 'is_event_chunk': False} for _ in doc]
    for chunk in doc.noun_chunks:
        has_event_keyword = any(token.text.lower() in feature_extractor.event_words for token in chunk)
        if len(chunk) == 1:
            chunk_info[chunk.start]['tag'] = 'S-CHUNK'
        else:
            chunk_info[chunk.start]['tag'] = 'B-CHUNK'
            for i in range(chunk.start + 1, chunk.end - 1):
                chunk_info[i]['tag'] = 'I-CHUNK'
            chunk_info[chunk.end - 1]['tag'] = 'E-CHUNK'
        if has_event_keyword:
            for i in range(chunk.start, chunk.end):
                chunk_info[i]['is_event_chunk'] = True

    return [feature_extractor.enhanced_word_features(doc, i, chunk_info) for i in range(len(doc))]

# --- 3. Main Processing Function ---
def process_text_xp(text: str) -> dict:
    print("\n" + "="*20 + " XP MODEL DEBUG START " + "="*20)
    print(f"1. INITIAL INPUT: '{text}'")

    if not crf_model:
        # ... (error handling is the same)
        return {"error": "CRF model is not loaded."}

    # Stage 1: Text Cleaning
    cleaned_text = spell_checker.correct_text_smart(text, learn=False, verbose=False)
    print(f"2. AFTER TEXT CLEANER: '{cleaned_text}'")

    # Stage 2: NER Prediction
    doc = nlp(cleaned_text)
    tokens = [token.text for token in doc]
    features = sent2features(tokens)
    predicted_bio_labels = crf_model.predict([features])[0]
    print("3. AFTER NER PREDICTION (CRF MODEL OUTPUT):")
    for token, label in zip(tokens, predicted_bio_labels):
        print(f"   - {token:<15} -> {label}")

    # --- Stage 3: ENHANCED Structuring and Post-Processing ---
    # 1. Initialize our new, smarter post-processor
    post_processor = EnhancedEventExtractor()
    
    # 2. Use its method to convert BIO tags to a basic dictionary
    # This replaces the old bio_to_structured function
    structured_data = post_processor.bio_to_structured_enhanced(tokens, predicted_bio_labels)
    print(f"4. AFTER INITIAL STRUCTURING: {structured_data}")
    
    # 3. Apply the final post-processing rules (regex, cleaning, etc.)
    # This replaces the old refine_event_entities function
    refined_entities = post_processor.post_process_events(structured_data, cleaned_text)
    print(f"5. AFTER FINAL REFINEMENT: {refined_entities}")

    # --- Stage 4: Normalization (using the refined entities) ---
    print("6. NORMALIZATION STAGE:")
    final_output = {"event_title": None, "date": None, "time": None}
    
    # Use .get() to safely access keys that might not exist
    if refined_entities.get("EVENT"):
        final_output["event_title"] = " ".join(refined_entities["EVENT"])

    if refined_entities.get("DATE"):
        date_text = " ".join(refined_entities["DATE"])
        normalized_date = normalize_temporal_expression(date_text, "DATE")
        final_output["date"] = f"{date_text} ({normalized_date})"
        print(f"   - Normalizing DATE: '{date_text}' -> '{normalized_date}'")

    if refined_entities.get("TIME"):
        time_text = " ".join(refined_entities["TIME"])
        normalized_time = normalize_temporal_expression(time_text, "TIME")
        final_output["time"] = f"{time_text} ({normalized_time})"
        print(f"   - Normalizing TIME: '{time_text}' -> '{normalized_time}'")

    print(f"7. FINAL STRUCTURED DATA: {final_output}")
    print("="*22 + " XP MODEL DEBUG END " + "="*22 + "\n")

    return {"structured_data": final_output}
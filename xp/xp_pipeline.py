# xp/xp_pipeline.py
import os
import re
import spacy
import joblib
from datetime import datetime

from .xp_post_processor import EnhancedEventExtractor
from .temporal_normalizer import normalize_temporal_expression
from .text_cleaner import SmartSpellChecker
from .xp_features import FeatureExtractor

# --- 1. Load Models and Tools ---
feature_extractor = FeatureExtractor()
spell_checker = SmartSpellChecker()
nlp = spacy.load("en_core_web_sm")

script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(script_dir, "xp_crf_model.joblib")
print(f"Loading CRF model from: {MODEL_PATH}")
try:
    crf_model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    print(f"FATAL ERROR: Model not found at {MODEL_PATH}")
    crf_model = None

def sent2features(sent_tokens):
    """Convert sentence tokens to features."""
    if not sent_tokens:
        return []
    doc = nlp(" ".join(sent_tokens))
    if len(doc) != len(sent_tokens):
        return []
    chunk_info = [{"tag": "O", "is_event_chunk": False} for _ in doc]
    for chunk in doc.noun_chunks:
        has_event_keyword = any(token.text.lower() in feature_extractor.event_words for token in chunk)
        if len(chunk) == 1:
            chunk_info[chunk.start]["tag"] = "S-CHUNK"
        else:
            chunk_info[chunk.start]["tag"] = "B-CHUNK"
            for i in range(chunk.start + 1, chunk.end - 1):
                chunk_info[i]["tag"] = "I-CHUNK"
            chunk_info[chunk.end - 1]["tag"] = "E-CHUNK"
        if has_event_keyword:
            for i in range(chunk.start, chunk.end):
                chunk_info[i]["is_event_chunk"] = True
    return [feature_extractor.enhanced_word_features(doc, i, chunk_info) for i in range(len(doc))]

def merge_temporal_entities(structured_list):
    """
    Merges adjacent DATE and TIME entities into a single DATE_TIME entity.
    Expects a list of dictionaries like [{'type': 'DATE', 'text': 'today'}, {'type': 'TIME', 'text': 'afternoon'}]
    """
    merged_list = []
    i = 0
    while i < len(structured_list):
        entity = structured_list[i]
        
        # Check for a DATE followed by a TIME
        if entity['type'] == 'DATE' and i + 1 < len(structured_list) and structured_list[i+1]['type'] == 'TIME':
            combined_text = f"{entity['text']} {structured_list[i+1]['text']}"
            merged_list.append({'type': 'DATE_TIME', 'text': combined_text})
            i += 2  # Skip both entities
        else:
            merged_list.append(entity)
            i += 1
            
    # Convert back to dictionary format for the rest of the pipeline
    final_dict = {}
    for entity in merged_list:
        if entity['type'] not in final_dict:
            final_dict[entity['type']] = []
        final_dict[entity['type']].append(entity['text'])
        
    return final_dict

def process_text_xp(text: str) -> dict:
    print("\n" + "=" * 20 + " XP MODEL DEBUG START " + "=" * 20)
    print(f"1. INITIAL INPUT: '{text}'")
    if not crf_model:
        return {"error": "CRF model is not loaded."}
    
    # Stage 1: Text Cleaning
    cleaned_text = spell_checker.correct_text_smart(text, learn=False, verbose=False)
    print(f"2. AFTER TEXT CLEANER: '{cleaned_text}'")
    
    # Stage 2: NER Prediction
    doc = nlp(cleaned_text)
    tokens = [token.text for token in doc]
    features = sent2features(tokens)
    predicted_bio_labels = crf_model.predict([features])[0]
    print("3. AFTER NER PREDICTION:")
    for token, label in zip(tokens, predicted_bio_labels):
        print(f"   - {token:<15} -> {label}")
    
    # Stage 3: Structuring and Merging
    post_processor = EnhancedEventExtractor()
    structured_list = post_processor.bio_to_structured_enhanced(tokens, predicted_bio_labels, return_list=True)
    print(f"4. AFTER INITIAL STRUCTURING: {structured_list}")
    
    refined_entities = merge_temporal_entities(structured_list)
    print(f"5. AFTER ENTITY MERGING: {refined_entities}")
    
    # Stage 4: Normalization
    print("6. NORMALIZATION STAGE:")
    final_output = {"event_title": None, "date": None, "time": None}

    if refined_entities.get("EVENT"):
        final_output["event_title"] = " ".join(refined_entities["EVENT"])

    if refined_entities.get("DATE_TIME"):
        date_time_text = refined_entities["DATE_TIME"][0]
        # This is where we use the new `DATE_TIME` normalizer type
        normalized_value = normalize_temporal_expression(date_time_text, "DATE_TIME")
        
        if normalized_value:
            normalized_date = normalized_value.split('T')[0]
            normalized_time = 'T' + normalized_value.split('T')[1]
            final_output["date"] = f"{date_time_text} ({normalized_date})"
            final_output["time"] = f"{date_time_text} ({normalized_time})"
            print(f"   - Normalizing DATE_TIME: '{date_time_text}' -> '{normalized_value}'")

    elif refined_entities.get("DATE"):
        date_text = refined_entities["DATE"][0]
        normalized_date = normalize_temporal_expression(date_text, "DATE")
        final_output["date"] = f"{date_text} ({normalized_date})"
        print(f"   - Normalizing DATE: '{date_text}' -> '{normalized_date}'")

    if refined_entities.get("TIME") and not refined_entities.get("DATE_TIME"):
        time_text = refined_entities["TIME"][0]
        normalized_time = normalize_temporal_expression(time_text, "TIME")
        final_output["time"] = f"{time_text} ({normalized_time})"
        print(f"   - Normalizing TIME: '{time_text}' -> '{normalized_time}'")

    print(f"7. FINAL STRUCTURED DATA: {final_output}")
    print("=" * 22 + " XP MODEL DEBUG END " + "=" * 22 + "\n")

    return {"structured_data": final_output}
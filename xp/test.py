import joblib
import os
import spacy

# Define the path where your model should be saved
# Make sure this path is identical to the one in your training script
MODEL_PATH = "xp/xp_crf_model.joblib" 

print(f"Attempting to load model from: {MODEL_PATH}")

try:
    crf_model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: Model file not found at the specified path.")
    exit()

# Simple test function
def test_model_on_word(model, word):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(word)
    # The simplest features to make a prediction
    test_features = [{"word": token.text.lower()} for token in doc]
    predicted_labels = model.predict([test_features])[0]
    return predicted_labels[0]

# Run the test
print("\n--- Testing Model Prediction ---")
prediction = test_model_on_word(crf_model, "tomorrow")
print(f"Prediction for 'tomorrow': {prediction}")
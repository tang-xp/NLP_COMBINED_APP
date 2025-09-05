# app.py

from flask import Flask, request, jsonify, render_template
import traceback # Import traceback for detailed error logging

# ==============================================================================
# INTEGRATION BLOCK - Import each member's main function here
# ==============================================================================

# --- XP's Model Import ---
try:
    from xp.xp_pipeline import process_text_xp
except ImportError:
    print("WARNING: Could not import xp_pipeline. Is the file structure correct?")
    def process_text_xp(text): return {"error": "XP's model pipeline failed to load."}

# --- Teammate A's Model Import (EXAMPLE) ---
# This will fail until teammate_A/bert_pipeline.py is created
try:
    # from teammate_A.bert_pipeline import process_text_A
    def process_text_A(text): return {"structured_data": {"event_title": "Example Event A", "date": "Tomorrow", "time": "10am"}} # Placeholder
except ImportError:
    def process_text_A(text): return {"error": "Teammate A's model is not available."}

# --- Teammate B's Model Import (EXAMPLE) ---
# This will fail until teammate_B/bayes_pipeline.py is created
try:
    # from teammate_B.bayes_pipeline import process_text_B
    def process_text_B(text): return {"structured_data": {"event_title": "Example Event B", "date": "Next Friday"}} # Placeholder
except ImportError:
    def process_text_B(text): return {"error": "Teammate B's model is not available."}

# ==============================================================================
# FLASK APPLICATION LOGIC - No need to edit below here
# ==============================================================================

app = Flask(__name__)

@app.route('/')
def home():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/process-text', methods=['POST'])
def process_text():
    """
    Receives text from the frontend, runs it through all models,
    and returns a combined JSON response.
    """
    data = request.get_json()
    text = data.get("message", "")
    if not text:
        return jsonify({"error": "No text provided."}), 400

    # --- Call each model's pipeline ---
    # Use try/except blocks to prevent one model's error from crashing the app
    all_results = {}

    # --- XP's Model Execution ---
    try:
        all_results['xp_model'] = process_text_xp(text)
    except Exception as e:
        print(f"ERROR in XP's model: {e}")
        traceback.print_exc() # Print full error trace to the console
        all_results['xp_model'] = {'error': "An internal error occurred in the CRF model."}

    # --- Teammate A's Model Execution ---
    try:
        all_results['teammate_A_model'] = process_text_A(text)
    except Exception as e:
        print(f"ERROR in Teammate A's model: {e}")
        all_results['teammate_A_model'] = {'error': "An internal error occurred in Teammate A's model."}

    # --- Teammate B's Model Execution ---
    try:
        all_results['teammate_B_model'] = process_text_B(text)
    except Exception as e:
        print(f"ERROR in Teammate B's model: {e}")
        all_results['teammate_B_model'] = {'error': "An internal error occurred in Teammate B's model."}


    return jsonify(all_results)

@app.route('/xp_normalize_text', methods=['POST'])
def xp_normalize_text():
    """
    Receives text and type, and normalizes it using XP's specific logic.
    """
    # ... (the rest of the function logic is exactly the same)
    data = request.get_json()
    text_to_normalize = data.get("text", "")
    entity_type = data.get("type", "").upper()

    if not text_to_normalize or not entity_type:
        return jsonify({"error": "Missing text or type for normalization."}), 400

    try:
        from xp.temporal_normalizer import normalize_temporal_expression
        normalized_value = normalize_temporal_expression(text_to_normalize, entity_type)
        full_normalized_string = f"{text_to_normalize} ({normalized_value})"
        return jsonify({"normalized_text": full_normalized_string})
    except Exception as e:
        print(f"ERROR in XP normalization endpoint: {e}")
        return jsonify({"error": "Failed to normalize text."}), 500

if __name__ == '__main__':
    app.run(debug=True)
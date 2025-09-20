# ==============================================================================
# app.py - Main Flask Application with Enhanced Error Handling
# ==============================================================================

from flask import Flask, request, jsonify, render_template
import traceback

# --- 1. Import your project's specific functions ---
# These are the only functions the web server needs to know about.
from xp.xp_pipeline import process_text_xp
from xp.temporal_normalizer import normalize_temporal_expression


# --- 2. Initialize the Flask App ---
app = Flask(__name__)


# --- 3. Define the route for the main homepage ---
# This serves your index.html file to the user's browser.
@app.route('/')
def home():
    """Serves the main HTML page."""
    return render_template('index.html')

# ------------------------------------------------------------------------------

# --- 4. Define the route for processing a user's initial message ---
@app.route('/process-text', methods=['POST'])
def process_text():
    """
    Receives text from the user, runs it through your CRF model pipeline,
    and returns the structured result.
    """
    try:
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"error": "Invalid request format."}), 400
        
        text = data.get("message", "")
        if not text:
            return jsonify({"error": "No text provided."}), 400

        # Call your main pipeline function inside the try block
        xp_result = process_text_xp(text)
        
        # The frontend is designed to look for a key named "xp_model",
        # so we wrap the result in that structure.
        return jsonify({"xp_model": xp_result})

    except Exception as e:
        # A general catch-all for any unexpected errors from the pipeline
        print(f"An unexpected error occurred: {e}")
        # Use traceback to get a full stack trace for detailed debugging
        traceback.print_exc()
        
        # Send a user-friendly error back to the frontend with a 500 status code
        return jsonify({
            'xp_model': {
                'structured_data': {
                    "event_title": None,
                    "date": "None",
                    "time": "None"
                },
                'error': f"An internal server error occurred: {str(e)}. Please check the server logs for details."
            }
        }), 500

# ------------------------------------------------------------------------------

# --- 5. Define the route for your model's specific normalization logic ---
@app.route('/xp_normalize_text', methods=['POST'])
def xp_normalize_text():
    """
    Called by the frontend to normalize a single piece of user-provided
    information (like a date or time).
    """
    try:
        data = request.get_json()
        text_to_normalize = data.get("text", "")
        entity_type = data.get("type", "").upper()

        if not text_to_normalize or not entity_type:
            return jsonify({"error": "Missing text or type for normalization."}), 400

        # Call your normalization function.
        # This is where the ValueError bug needs to be fixed.
        normalized_value = normalize_temporal_expression(text_to_normalize, entity_type)
        
        # Check if the normalization was successful before formatting.
        if normalized_value:
            full_normalized_string = f"{text_to_normalize} ({normalized_value})"
            return jsonify({"normalized_text": full_normalized_string})
        else:
            return jsonify({"error": "Normalization failed for the provided text."}), 400

    except ValueError as ve:
        # Catch a specific ValueError from the normalizer
        print(f"Normalization ValueError: {ve}")
        return jsonify({
            "error": "I couldn't normalize that expression. Please rephrase it.",
            "details": str(ve)
        }), 400 # Return a 400 Bad Request status code
    
    except Exception as e:
        print(f"ERROR in normalization endpoint: {e}")
        traceback.print_exc()
        return jsonify({'error': "Failed to normalize text due to a server error."}), 500

# ------------------------------------------------------------------------------

# --- 6. Run the App ---
if __name__ == '__main__':
    # The 'debug=True' flag means the server will automatically reload
    # if you save changes to this file.
    app.run(debug=True)
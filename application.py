import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Absolute paths for reliability
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "artifacts/model/model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "artifacts/processed/preprocessor.pkl")

def load_ml_assets():
    """Load pre-trained artifacts once during server startup."""
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        return model, preprocessor
    except Exception as e:
        app.logger.error(f"Critical asset load failure: {e}")
        return None, None

MODEL, PREPROCESSOR = load_ml_assets()

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Real-time inference endpoint with rigorous type validation."""
    if not MODEL or not PREPROCESSOR:
        return jsonify({"error": "Assets missing"}), 500

    try:
        # Explicit mapping to match the Top 10 MI features exactly
        # Using .get() ensures no KeyErrors; defaults prevent 'isnan' errors
        raw_payload = {
            'use_chip': str(request.form.get('use_chip', 'Chip Transaction')),
            'merchant_state': str(request.form.get('merchant_state', 'UNKNOWN')),
            'card_brand': str(request.form.get('card_brand', 'Visa')),
            'card_type': str(request.form.get('card_type', 'Credit')),
            'has_chip': int(request.form.get('has_chip', 1)),
            'num_cards_issued': int(request.form.get('num_cards_issued', 1)),
            'gender': str(request.form.get('gender', 'Male')),
            'txn_dayofweek': int(request.form.get('txn_dayofweek', 0)),
            'days_since_pin_change': int(request.form.get('days_since_pin_change', 0)),
            'age_bucket': str(request.form.get('age_bucket', '25-40'))
        }

        # Transform to DataFrame to preserve feature metadata
        df_input = pd.DataFrame([raw_payload])

        # Step 1: Preprocessing (OneHot + Scaling)
        # This isolates categorical strings from the numerical scaler
        feature_vector = PREPROCESSOR.transform(df_input)

        # Step 2: Scoring
        prob_score = MODEL.predict_proba(feature_vector)[0][1]
        
        # Decision boundary (0.50)
        result = "Fraudulent" if prob_score > 0.5 else "Legitimate"

        return render_template(
            "index.html",
            prediction=result,
            probability=f"{prob_score:.2%}",
            status_class="danger" if result == "Fraudulent" else "success"
        )

    except Exception as e:
        app.logger.error(f"Predict error details: {str(e)}")
        # Returning detail helps identify which specific field failed
        return jsonify({"error": f"Inference failed: {str(e)}"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
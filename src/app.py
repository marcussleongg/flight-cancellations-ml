from flask import Flask, request, jsonify, render_template
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import sys

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import preprocessing function from predict.py
from predict import preprocess_new_data

app = Flask(__name__)

# Load model and preprocessing artifacts (load once at startup)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODELS_DIR / "random_forest.pkl"
PREPROC_PATH = MODELS_DIR / "preprocessing.pkl"

rf_model = joblib.load(MODEL_PATH)
preproc = joblib.load(PREPROC_PATH)

def _is_valid_hhmm(x) -> bool:
    try:
        v = int(x)
    except (TypeError, ValueError):
        return False
    if v < 0 or v > 2359:
        return False
    mm = v % 100
    hh = v // 100
    return 0 <= hh <= 23 and 0 <= mm <= 59

def _validate_payload(data: dict) -> list[str]:
    errors: list[str] = []

    required = ["AIRLINE", "FL_DATE", "ORIGIN", "DEST", "CRS_DEP_TIME", "CRS_ARR_TIME"]
    for k in required:
        if k not in data or data[k] in (None, ""):
            errors.append(f"Missing required field: {k}")

    # Airport codes
    for k in ["ORIGIN", "DEST"]:
        if k in data and isinstance(data[k], str):
            code = data[k].strip().upper()
            if len(code) != 3 or not code.isalpha():
                errors.append(f"{k} must be a 3-letter airport code (e.g., JFK)")
        elif k in data and data.get(k) not in (None, ""):
            errors.append(f"{k} must be a string airport code (e.g., JFK)")

    # Times
    if "CRS_DEP_TIME" in data and data.get("CRS_DEP_TIME") not in (None, "") and not _is_valid_hhmm(data["CRS_DEP_TIME"]):
        errors.append("CRS_DEP_TIME must be HHMM in 24-hour time (0000–2359, minutes < 60)")
    if "CRS_ARR_TIME" in data and data.get("CRS_ARR_TIME") not in (None, "") and not _is_valid_hhmm(data["CRS_ARR_TIME"]):
        errors.append("CRS_ARR_TIME must be HHMM in 24-hour time (0000–2359, minutes < 60)")

    # Date parse
    if "FL_DATE" in data and data.get("FL_DATE") not in (None, ""):
        try:
            pd.to_datetime(data["FL_DATE"])
        except Exception:
            errors.append("FL_DATE must be a valid date (YYYY-MM-DD)")

    return errors

@app.route('/')
def index():
    """Serve the frontend HTML page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for making predictions"""
    try:
        # Get JSON data from frontend
        data = request.json or {}

        # Validate user input (backend safety)
        errors = _validate_payload(data)
        if errors:
            return jsonify({"success": False, "error": "; ".join(errors)}), 400
        
        # Convert to DataFrame (single row)
        df_raw = pd.DataFrame([data])
        
        # Preprocess
        X_new = preprocess_new_data(df_raw)
        
        # Make prediction
        y_proba = rf_model.predict_proba(X_new)[:, 1][0]
        y_pred = rf_model.predict(X_new)[0]
        
        # Return result
        return jsonify({
            'success': True,
            'prediction': int(y_pred),
            'probability': float(y_proba),
            'message': 'Flight will be cancelled' if y_pred == 1 else 'Flight will not be cancelled'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

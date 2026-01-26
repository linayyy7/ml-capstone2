"""
Prediction service for Remote Worker Productivity Prediction

This script:
- Loads trained model and DictVectorizer from model.bin
- Exposes a REST API using Flask
- Accepts raw feature input as JSON
- Applies the same preprocessing as training
- Returns predicted productivity_score
"""

import pickle
from flask import Flask, request, jsonify


# ================================
# Configuration
# ================================

MODEL_PATH = "model.bin"


# ================================
# Load model bundle
# ================================

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
dv = bundle["dv"]


# ================================
# Flask app
# ================================

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict productivity score for a single worker.

    Expected JSON payload:
    {
        "location_type": "...",
        "industry_sector": "...",
        "age": ...,
        "experience_years": ...,
        "average_daily_work_hours": ...,
        "break_frequency_per_day": ...,
        "late_task_ratio": ...,
        "calendar_scheduled_usage": ...,
        "focus_time_minutes": ...,
        "tool_usage_frequency": ...,
        "automated_task_count": ...,
        "AI_assisted_planning": ...,
        "real_time_feedback_score": ...
    }
    """

    worker = request.get_json()

    if worker is None:
        return jsonify({"error": "Invalid or missing JSON payload"}), 400

    # DictVectorizer expects a list of dictionaries
    worker_dict = [worker]

    # Apply the same encoding as training
    X = dv.transform(worker_dict)

    # Generate prediction
    prediction = model.predict(X)[0]

    return jsonify({
        "predicted_productivity_score": float(prediction)
    })


# ================================
# Entry point
# ================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696, debug=True)

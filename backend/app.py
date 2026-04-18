from flask import Flask, request, jsonify, redirect
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import os

from nlp_processor import expand_symptoms

# -------------------------------------------------
# APP SETUP
# -------------------------------------------------
app = Flask(__name__)
CORS(app)

CONFIDENCE_THRESHOLD = 0.40

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
MODEL_PATH = "model_files/disease_model.pkl"
DATASET_PATH = "dataset/Training.csv"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found")

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
vectorizer = bundle["vectorizer"]

# -------------------------------------------------
# LOAD METADATA
# -------------------------------------------------
df = pd.read_csv(DATASET_PATH)
df["Disease"] = df["Disease"].str.lower().str.strip()

disease_meta = {}
for _, row in df.iterrows():
    disease_meta[row["Disease"]] = {
        "severity_score": int(row.get("Severity_Score", 5)),
        "doctor_category": row.get("Doctor_Category", "General Physician"),
        "description": row.get("Description", ""),
        "prescription": row.get("Prescription", ""),
        "tests": row.get("Tests", ""),
        "precautions": row.get("Precautions", "")
    }

# -------------------------------------------------
# HEALTH CHECK
# -------------------------------------------------
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "ok",
        "message": "AI Doctor Backend Running"
    })

# -------------------------------------------------
# PREDICTION WITH EXPLAINABLE AI
# -------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(silent=True)
    if not body or "message" not in body:
        return jsonify({"status": "error", "message": "Invalid request"}), 400

    user_text = body["message"]
    processed_text = expand_symptoms(user_text)

    if not processed_text.strip():
        return jsonify({"status": "low_confidence"})

    X = vectorizer.transform([processed_text])
    probabilities = model.predict_proba(X)[0]

    top_indices = np.argsort(probabilities)[-3:][::-1]

    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = X.toarray()[0]

    results = []

    for idx in top_indices:
        confidence = float(probabilities[idx])
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        disease = model.classes_[idx]
        meta = disease_meta.get(disease, {})

        # 🔍 Explainable AI – symptoms that matched user input
        explanation = [
            feature_names[i]
            for i in tfidf_scores.argsort()[::-1]
            if tfidf_scores[i] > 0
        ][:5]

        results.append({
            "disease": disease.title(),
            "confidence": round(confidence * 100, 2),
            "why_prediction": explanation,
            "severity_score": meta.get("severity_score", 5),
            "doctor_category": meta.get("doctor_category"),
            "description": meta.get("description"),
            "prescription": meta.get("prescription"),
            "tests": meta.get("tests"),
            "precautions": meta.get("precautions")
        })

    if not results:
        return jsonify({"status": "low_confidence"})

    return jsonify({
        "status": "success",
        "top_predictions": results
    })

# -------------------------------------------------
# REAL VIDEO DOCTOR CALL (WORKING)
# -------------------------------------------------
@app.route("/doctor-call", methods=["GET"])
def doctor_call():
    return redirect("https://meet.jit.si/AI_Doctor_Consultation")

# -------------------------------------------------
# ENTRY POINT (RENDER / RAILWAY READY)
# -------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)


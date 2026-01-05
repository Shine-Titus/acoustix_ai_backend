import joblib
import numpy as np
from feature_extraction import extract_features

MODEL_PATH = "model/isolation_forest.pkl"
THRESHOLD = -0.02   

model = joblib.load(MODEL_PATH)


def score_to_confidence(score, threshold):
    if score >= threshold:
        return min(100, 60 + (score - threshold) * 200)
    else:
        return max(0, 60 - (threshold - score) * 200)


def infer_audio(audio_path):
    features = extract_features(audio_path)
    score = model.decision_function([features])[0]

    is_anomaly = score < THRESHOLD
    confidence = score_to_confidence(score, THRESHOLD)

    return {
        "status": "Anomalous" if is_anomaly else "Normal",
        "features": list(features), 
        "confidence": round(confidence, 2),
        "score": float(score) 
    }

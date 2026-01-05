import os
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib

from feature_extraction import extract_features

DATA_DIR = "training_data"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "isolation_forest.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

X_train = []

for file in os.listdir(DATA_DIR):
    if file.endswith(".wav") or file.endswith(".mp3"):
        file_path = os.path.join(DATA_DIR, file)
        features = extract_features(file_path)
        X_train.append(features)

X_train = np.array(X_train)

print("Training data shape:", X_train.shape)

model = IsolationForest(
    n_estimators=100,
    contamination=0.1,
    random_state=42
)

model.fit(X_train)

scores = model.decision_function(X_train)

for i, score in enumerate(scores):
    print(f"Sample {i}: score = {score:.4f}")

joblib.dump(model, MODEL_PATH)
print("Model saved to:", MODEL_PATH)




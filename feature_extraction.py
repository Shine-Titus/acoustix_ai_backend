import librosa
import numpy as np
from pathlib import Path


def load_and_preprocess_audio(file_path, sr=16000):
    # Load audio
    y, _ = librosa.load(file_path, sr=sr, mono=True)

    # Trim leading & trailing silence
    y, _ = librosa.effects.trim(y, top_db=20)

    # Normalize amplitude
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    return y, sr

def aggregate_feature(feature):
    return np.array([
        np.mean(feature),
        np.std(feature)
    ])


def extract_features(file_path):
    y, sr = load_and_preprocess_audio(file_path)

    features = []

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(mfcc.shape[0]):
        features.extend(aggregate_feature(mfcc[i]))

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.extend(aggregate_feature(centroid))

    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.extend(aggregate_feature(rolloff))

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features.extend(aggregate_feature(zcr))

    return np.array(features)

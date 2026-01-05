import librosa
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def load_and_preprocess_audio(file_path, sr=16000):

    y, s = librosa.load(file_path, sr=sr, mono=True)

    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=s)
    plt.show()

    y, _ = librosa.effects.trim(y, top_db=20)

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

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(mfcc.shape[0]):
        features.extend(aggregate_feature(mfcc[i]))

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.extend(aggregate_feature(centroid))

    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.extend(aggregate_feature(rolloff))

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features.extend(aggregate_feature(zcr))

    return np.array(features)

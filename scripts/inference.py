"""
Inference CLI Tool for Tamil Speech Emotion Recognition
=========================================================
Predicts speech emotion class for a given raw input speech file (.wav)
using the pre-trained Support Vector Machine (SVM) model.

Reference Paper:
  Gokul Ram K, Vignesh U, & Shyam Karthinathan P K (2026). 
  Innovative Feature Fusion and XAI Framework for Robust Tamil Speech Emotion Recognition. 
  In IEEE ICIRCA 2026 (pp. 983-989). IEEE.
"""

import os
import sys
import argparse
import joblib
import numpy as np
from extract_features import extract_features

# Emotion label map
CLASSES = ['angry', 'fear', 'happy', 'neutral', 'sad']

def predict_emotion(audio_path, model_path="saved_models/svm_model_best.pkl", scaler_path=None):
    """Predicts emotion label and confidence scores for an audio file."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Extract 42D fused features
    print(f"🎙️ Extracting 42D fused acoustic features from: {audio_path}")
    feats = extract_features(audio_path).reshape(1, -1)

    # Load model
    model = joblib.load(model_path)

    # Scale features if scaler exists or model requires scaling
    scaler_file = scaler_path or os.path.join(os.path.dirname(model_path), "scaler.pkl")
    if os.path.exists(scaler_file):
        scaler = joblib.load(scaler_file)
        feats = scaler.transform(feats)

    # Predict
    pred_idx = model.predict(feats)[0]
    predicted_label = CLASSES[pred_idx] if pred_idx < len(CLASSES) else str(pred_idx)

    # Get probability distribution if supported
    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feats)[0]

    print("\n" + "="*50)
    print(f"🎯 PREDICTED EMOTION : {predicted_label.upper()}")
    print("="*50)
    if probs is not None:
        print("Probability Distribution:")
        for cls_name, prob in zip(CLASSES, probs):
            bar = "█" * int(prob * 30)
            print(f"  {cls_name:<10}: {prob*100:6.2f}% {bar}")
    print("="*50)

    return predicted_label, probs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tamil Speech Emotion Recognition Inference CLI")
    parser.add_argument("--audio", type=str, required=True, help="Path to input .wav audio file")
    parser.add_argument("--model", type=str, default="saved_models/svm_model_best.pkl", help="Path to pre-trained model pickle")
    
    args = parser.parse_args()
    predict_emotion(args.audio, args.model)

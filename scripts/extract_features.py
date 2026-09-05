"""
Audio Feature Extraction Pipeline for Tamil Speech Emotion Recognition (SER)
=============================================================================
This script extracts a 42-dimensional fused acoustic feature vector from input speech 
audio signals (.wav), combining:
  - 40 Mel-Frequency Cepstral Coefficients (MFCCs) [Timbral/Spectral]
  - Fundamental Pitch Track Mean via librosa piptrack [Prosodic/Pitch]
  - Root Mean Square (RMS) Energy Mean [Intensity/Dynamics]

Reference Paper:
  Gokul Ram K, Vignesh U, & Shyam Karthinathan P K (2026). 
  Innovative Feature Fusion and XAI Framework for Robust Tamil Speech Emotion Recognition. 
  In IEEE ICIRCA 2026 (pp. 983-989). IEEE.
"""

import os
import argparse
import numpy as np
import pandas as pd
import librosa

def extract_features(filepath, sr=16000, n_mfcc=40):
    """
    Extracts fused acoustic features from an audio file.

    Parameters:
        filepath (str): Path to audio file (.wav).
        sr (int): Target sampling rate (default: 16000 Hz).
        n_mfcc (int): Number of MFCC coefficients to extract (default: 40).

    Returns:
        np.ndarray: 42-dimensional feature vector [MFCC_0...MFCC_39, Pitch_Mean, RMS_Energy_Mean].
    """
    try:
        y, sr = librosa.load(filepath, sr=sr)

        # 1. 40 MFCCs (Time-averaged mean across frames)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs, axis=1)

        # 2. Fundamental Pitch Track Mean
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch = pitches[magnitudes > np.median(magnitudes)]
        pitch_mean = np.mean(pitch) if len(pitch) > 0 else 0.0

        # 3. RMS Energy Mean
        rms = librosa.feature.rms(y=y)
        energy_mean = np.mean(rms)

        # Fuse into a 42-dimensional feature vector
        return np.hstack([mfccs_mean, pitch_mean, energy_mean])
    except Exception as e:
        print(f"Error extracting features from {filepath}: {e}")
        return np.zeros(n_mfcc + 2)

def get_feature_names(n_mfcc=40):
    """Returns descriptive column names for the 42 extracted features."""
    return [f"MFCC_{i}" for i in range(n_mfcc)] + ["Pitch_Mean", "RMS_Energy_Mean"]

def process_dataset(csv_path, output_npz=None):
    """Processes an entire dataset CSV with file paths and labels."""
    df = pd.read_csv(csv_path)
    X, y = [], []
    
    print(f"Extracting features for {len(df)} audio samples...")
    for idx, row in df.iterrows():
        fpath = row['filepath']
        feats = extract_features(fpath)
        X.append(feats)
        y.append(row['label'])
        if (idx + 1) % 100 == 0 or (idx + 1) == len(df):
            print(f"  Processed {idx + 1}/{len(df)} samples...")
            
    X = np.array(X)
    y = np.array(y)
    
    if output_npz:
        np.savez(output_npz, X=X, y=y)
        print(f"Features saved to {output_npz}")
        
    return X, y

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 42D Fused Features for Tamil SER")
    parser.add_argument("--file", type=str, help="Path to single .wav file")
    parser.add_argument("--csv", type=str, help="Path to dataset metadata CSV")
    parser.add_argument("--output", type=str, default="features.npz", help="Output .npz path for dataset batch mode")
    
    args = parser.parse_args()
    
    if args.file:
        feats = extract_features(args.file)
        print(f"Extracted feature vector shape: {feats.shape}")
        print("Feature Vector Sample:", feats[:5])
    elif args.csv:
        process_dataset(args.csv, args.output)
    else:
        print("Please provide --file <audio.wav> or --csv <dataset.csv>")

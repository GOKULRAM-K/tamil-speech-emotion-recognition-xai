"""
Model Training and Evaluation Pipeline for Tamil Speech Emotion Recognition
=============================================================================
Trains and evaluates 8 Machine Learning classifiers on 42-dimensional fused audio features:
  1. Support Vector Machine (SVM) - Best Performing Model (70.2% Accuracy / 0.71 Macro F1)
  2. Extra Trees Classifier
  3. Random Forest Classifier
  4. XGBoost Classifier
  5. Gradient Boosting Classifier
  6. Logistic Regression
  7. Naive Bayes (GaussianNB)
  8. K-Nearest Neighbors (KNN)

Reference Paper:
  Gokul Ram K, Vignesh U, & Shyam Karthinathan P K (2026). 
  Innovative Feature Fusion and XAI Framework for Robust Tamil Speech Emotion Recognition. 
  In IEEE ICIRCA 2026 (pp. 983-989). IEEE.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

def train_and_evaluate(X, y, save_dir="../saved_models"):
    """
    Executes train-test split (80/20 stratified), scales data where appropriate,
    trains all 8 benchmark models, prints classification metrics, and exports pickles.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Encode target labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    classes = encoder.classes_
    print("Dataset Target Classes:", classes)

    # Stratified Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )

    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # Standard Scaling for distance/gradient based models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))
    joblib.dump(encoder, os.path.join(save_dir, "label_encoder.pkl"))

    # Define all 8 Classifiers
    models = {
        "SVM (RBF Kernel)": (SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42), True, "svm_model_best.pkl"),
        "Extra Trees": (ExtraTreesClassifier(n_estimators=200, random_state=42), False, "extra_trees_model.pkl"),
        "Random Forest": (RandomForestClassifier(n_estimators=200, random_state=42), False, "random_forest_model.pkl"),
        "XGBoost": (XGBClassifier(n_estimators=100, max_depth=7, learning_rate=0.2, subsample=0.8, colsample_bytree=1.0, random_state=42, eval_metric="mlogloss"), False, "xgboost_model.pkl"),
        "Gradient Boosting": (GradientBoostingClassifier(random_state=42), False, "gradient_boosting_model.pkl"),
        "Logistic Regression": (LogisticRegression(max_iter=500, class_weight="balanced", random_state=42), True, "logistic_regression_model.pkl"),
        "Naive Bayes": (GaussianNB(), True, "naive_bayes_model.pkl"),
        "KNN (k=5)": (KNeighborsClassifier(n_neighbors=5), True, "knn_model.pkl")
    }

    summary_metrics = []

    print("\n" + "="*70)
    print(f"{'Model Name':<25} | {'Accuracy':<10} | {'Macro F1':<10} | {'Weighted F1':<10}")
    print("="*70)

    for name, (model, use_scaled, pkl_name) in models.items():
        X_tr = X_train_scaled if use_scaled else X_train
        X_te = X_test_scaled if use_scaled else X_test

        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)

        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        weighted_f1 = f1_score(y_test, y_pred, average="weighted")

        summary_metrics.append({
            "Model": name,
            "Accuracy": acc,
            "Macro F1": macro_f1,
            "Weighted F1": weighted_f1
        })

        print(f"{name:<25} | {acc*100:6.2f}%    | {macro_f1:8.4f}   | {weighted_f1:11.4f}")

        # Save model pickle
        joblib.dump(model, os.path.join(save_dir, pkl_name))

    print("="*70)
    print(f"Pre-trained model binaries exported to {save_dir}")
    return pd.DataFrame(summary_metrics)

if __name__ == "__main__":
    print("Train & Eval pipeline ready. Load features array X and y to execute benchmark.")

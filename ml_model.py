"""
Standalone run:  python ml_model.py
"""

import os, sys
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

#Constants
MODEL_PATH    = "models/risk_classifier.pkl"
SCALER_PATH   = "models/scaler.pkl"
FEATURES_PATH = "models/features.pkl"
DATA_PATH     = "data/mental_health_dataset.csv"

FEATURES = [
    "age", "sleep_hours", "stress_level",
    "exercise_days_per_week", "social_support_score",
    "work_hours_per_week", "screen_time_hours",
]
RISK_LABELS = ["Minimal", "Mild", "Moderate", "Severe"]

#Train
def train():
    """Train and persist the model. Returns (model, scaler, accuracy)."""
    if not os.path.exists(DATA_PATH):
        sys.path.insert(0, "data")
        import generate_dataset  # noqa — generates the CSV as a side effect

    df = pd.read_csv(DATA_PATH)
    X  = df[FEATURES].values
    y  = df["risk_numeric"].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    # Use labels= to avoid crash when not all classes appear in test split
    report = classification_report(
        y_test, y_pred,
        labels=[0, 1, 2, 3],
        target_names=RISK_LABELS,
        zero_division=0,
    )
    print(f"Model trained - accuracy: {acc:.2%}\n{report}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model,    MODEL_PATH)
    joblib.dump(scaler,   SCALER_PATH)
    joblib.dump(FEATURES, FEATURES_PATH)
    print(f"✅ Saved → {MODEL_PATH}")
    return model, scaler, acc

#Load (cached)
def load():
    """Load model + scaler from disk, training first if missing."""
    if not os.path.exists(MODEL_PATH):
        train()
    model    = joblib.load(MODEL_PATH)
    scaler   = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, scaler, features

#Predict
def predict(age, sleep_hours, stress_level, exercise_days,
            social_support, work_hours, screen_time):
    """
    Returns (risk_label, probabilities_dict).
    Example:
        label, probs = predict(25, 6.0, 7, 2, 5, 45, 6)
    """
    model, scaler, _ = load()
    x = np.array([[age, sleep_hours, stress_level,
                   exercise_days, social_support,
                   work_hours, screen_time]])
    x_scaled = scaler.transform(x)
    cls      = model.predict(x_scaled)[0]
    proba    = model.predict_proba(x_scaled)[0]
    label    = RISK_LABELS[cls]
    probs    = {RISK_LABELS[i]: round(float(p) * 100, 1) for i, p in enumerate(proba)}
    return label, probs

#Evaluation helpers (used in app.py Tab 3)
def get_eval_data():
    """Returns y_test, y_pred, accuracy for confusion matrix display."""
    model, scaler, _ = load()
    df   = pd.read_csv(DATA_PATH)
    X    = scaler.transform(df[FEATURES].values)
    y    = df["risk_numeric"].values
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    y_pred = model.predict(X_test)
    return y_test, y_pred, accuracy_score(y_test, y_pred)

def get_feature_importance():
    """Returns sorted (feature_name, importance) list."""
    model, _, features = load()
    pairs = sorted(
        zip(features, model.feature_importances_),
        key=lambda x: x[1]
    )
    return pairs

if __name__ == "__main__":
    train()

"""
Train, evaluate, and save churn prediction models.
Compares Logistic Regression, Random Forest, and XGBoost.
Uses SMOTE for handling class imbalance.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Add parent dir to path so we can import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_preprocessing import preprocess_pipeline

warnings.filterwarnings("ignore")


# ─── Model Definitions ──────────────────────────────────────────────────────

MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42, class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, class_weight="balanced", n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        random_state=42, eval_metric="logloss",
        use_label_encoder=False,
    ),
}


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Evaluate a trained model and return metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    }

    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    print(f"ROC-AUC: {metrics['ROC-AUC']:.4f}")

    return metrics


# ─── Training Pipeline ──────────────────────────────────────────────────────

def train_all_models(X_train, X_test, y_train, y_test):
    """Train all models with SMOTE and return results."""

    # Apply SMOTE to training data only
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print(f"SMOTE: {len(y_train)} → {len(y_train_sm)} training samples")
    print(f"  Class distribution after SMOTE: {pd.Series(y_train_sm).value_counts().to_dict()}")

    results = []
    trained_models = {}

    for name, model in MODELS.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_sm, y_train_sm)
        metrics = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)
        trained_models[name] = model

    return pd.DataFrame(results), trained_models


def save_best_model(results_df, trained_models, scaler, feature_names, models_dir):
    """Save the best model (by ROC-AUC) along with scaler and feature names."""
    os.makedirs(models_dir, exist_ok=True)

    best_idx = results_df["ROC-AUC"].idxmax()
    best_name = results_df.loc[best_idx, "Model"]
    best_model = trained_models[best_name]

    # Save as a bundle: model + scaler + feature names
    bundle = {
        "model": best_model,
        "scaler": scaler,
        "feature_names": feature_names,
        "model_name": best_name,
    }

    save_path = os.path.join(models_dir, "churn_model.pkl")
    joblib.dump(bundle, save_path)
    print(f"\nBest model: {best_name} (ROC-AUC: {results_df.loc[best_idx, 'ROC-AUC']:.4f})")
    print(f"Saved → {save_path}")

    return best_model, best_name


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    raw_path = os.path.join(base_dir, "data", "raw data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    processed_path = os.path.join(base_dir, "data", "processed data", "processed_data.csv")
    models_dir = os.path.join(base_dir, "models")

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data not found at {raw_path}. Please add the Telco Customer Churn dataset.")

    # Preprocess
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_pipeline(
        raw_path, processed_path
    )

    # Train & evaluate all models
    results_df, trained_models = train_all_models(X_train, X_test, y_train, y_test)

    # Comparison table
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON")
    print("=" * 60)
    print(results_df.to_string(index=False, float_format="{:.4f}".format))

    # Save best model
    save_best_model(results_df, trained_models, scaler, feature_names, models_dir)

    return results_df, trained_models


if __name__ == "__main__":
    main()

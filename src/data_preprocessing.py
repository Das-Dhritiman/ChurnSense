"""
Data preprocessing pipeline for customer churn prediction.
Handles cleaning, encoding, feature engineering, and train-test splitting.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib


# ─── Columns ────────────────────────────────────────────────────────────────

CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

TARGET = "Churn"
DROP_COLS = ["customerID"]


# ─── Cleaning ───────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop IDs, fix types, handle missing values."""
    df = df.copy()

    # Drop customer ID (not a feature)
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    # TotalCharges can have blank strings in real data
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing TotalCharges with median
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    return df


# ─── Feature Engineering ────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create business-meaningful features."""
    df = df.copy()

    # Average monthly spend (captures value consistency)
    df["AvgMonthlySpend"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"]
    )

    # Tenure group (binned for interpretability)
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"]
    ).astype(str)

    # Number of services subscribed
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    df["NumServices"] = df[service_cols].apply(
        lambda row: sum(1 for v in row if v == "Yes"), axis=1
    )

    return df


# ─── Encoding ───────────────────────────────────────────────────────────────

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode binary columns, one-hot-encode multi-class columns."""
    df = df.copy()

    # Encode target
    if TARGET in df.columns:
        df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})

    # Binary columns → label encode
    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    for col in df.columns:
        if df[col].dtype == "object" and set(df[col].unique()).issubset(
            {"Yes", "No", "Male", "Female"}
        ):
            df[col] = df[col].map(binary_map)

    # Multi-class columns → one-hot encode
    multi_class_cols = [
        col for col in df.columns
        if df[col].dtype == "object" and col != TARGET
    ]
    df = pd.get_dummies(df, columns=multi_class_cols, drop_first=True)

    return df


# ─── Full Pipeline ──────────────────────────────────────────────────────────

def preprocess_pipeline(
    input_path: str,
    output_path: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    End-to-end preprocessing pipeline.

    Returns:
        X_train, X_test, y_train, y_test, scaler, feature_names
    """
    # Load
    df = pd.read_csv(input_path)

    # Clean
    df = clean_data(df)

    # Feature engineering
    df = engineer_features(df)

    # Encode
    df = encode_features(df)

    # Save processed data
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Processed data saved → {output_path}")

    # Split
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale numeric features
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges",
                        "AvgMonthlySpend", "NumServices"]
    numeric_features = [f for f in numeric_features if f in X_train.columns]

    scaler = StandardScaler()
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test[numeric_features] = scaler.transform(X_test[numeric_features])

    print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    print(f"Features: {X_train.shape[1]} | Churn rate: {y.mean():.1%}")

    return X_train, X_test, y_train, y_test, scaler, list(X.columns)


if __name__ == "__main__":
    base = os.path.join(os.path.dirname(__file__), "..")
    raw = os.path.join(base, "data", "raw data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    processed = os.path.join(base, "data", "processed data", "processed_data.csv")

    X_train, X_test, y_train, y_test, scaler, features = preprocess_pipeline(
        raw, processed
    )
    print(f"\nFeature names ({len(features)}):")
    print(features)

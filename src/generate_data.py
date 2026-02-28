"""
Generate realistic synthetic Telecom Customer Churn data.
Mirrors the structure of the classic Telco Customer Churn dataset.
"""

import os
import numpy as np
import pandas as pd

SEED = 42
N_CUSTOMERS = 7043  # Same size as the classic Telco dataset


def generate_churn_data(n: int = N_CUSTOMERS, seed: int = SEED) -> pd.DataFrame:
    """Generate a synthetic telecom churn dataset."""
    rng = np.random.RandomState(seed)

    customer_id = [f"CUST-{str(i).zfill(5)}" for i in range(1, n + 1)]
    gender = rng.choice(["Male", "Female"], n)
    senior_citizen = rng.choice([0, 1], n, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], n, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], n, p=[0.30, 0.70])

    # Tenure in months (1–72)
    tenure = rng.randint(1, 73, n)

    # Services
    phone_service = rng.choice(["Yes", "No"], n, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_service == "No", "No phone service",
        rng.choice(["Yes", "No"], n, p=[0.42, 0.58])
    )

    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22]
    )

    def _internet_addon(internet_service, rng, n, p_yes=0.35):
        return np.where(
            internet_service == "No", "No internet service",
            rng.choice(["Yes", "No"], n, p=[p_yes, 1 - p_yes])
        )

    online_security = _internet_addon(internet_service, rng, n, 0.29)
    online_backup = _internet_addon(internet_service, rng, n, 0.34)
    device_protection = _internet_addon(internet_service, rng, n, 0.34)
    tech_support = _internet_addon(internet_service, rng, n, 0.29)
    streaming_tv = _internet_addon(internet_service, rng, n, 0.38)
    streaming_movies = _internet_addon(internet_service, rng, n, 0.39)

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24]
    )
    paperless_billing = rng.choice(["Yes", "No"], n, p=[0.59, 0.41])
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)",
         "Credit card (automatic)"],
        n, p=[0.34, 0.23, 0.22, 0.21]
    )

    # Monthly charges — depends on services
    base_charge = 20.0
    monthly_charges = base_charge + rng.uniform(0, 10, n)
    monthly_charges += np.where(internet_service == "Fiber optic", 30, 0)
    monthly_charges += np.where(internet_service == "DSL", 15, 0)
    monthly_charges += np.where(online_security == "Yes", 5, 0)
    monthly_charges += np.where(online_backup == "Yes", 5, 0)
    monthly_charges += np.where(device_protection == "Yes", 5, 0)
    monthly_charges += np.where(tech_support == "Yes", 5, 0)
    monthly_charges += np.where(streaming_tv == "Yes", 10, 0)
    monthly_charges += np.where(streaming_movies == "Yes", 10, 0)
    monthly_charges += np.where(multiple_lines == "Yes", 5, 0)
    monthly_charges = np.round(monthly_charges + rng.normal(0, 3, n), 2)
    monthly_charges = np.clip(monthly_charges, 18.0, 120.0)

    total_charges = np.round(monthly_charges * tenure + rng.normal(0, 50, n), 2)
    total_charges = np.clip(total_charges, 18.0, None)

    # --- Churn label (realistic probabilistic model) ---
    # Use a log-odds (logistic) model for more realistic & learnable signal
    log_odds = np.full(n, -1.5)  # base ~18% churn

    # Contract type (strongest predictor)
    log_odds += np.where(contract == "Month-to-month", 1.6, 0)
    log_odds -= np.where(contract == "Two year", 1.2, 0)

    # Tenure (strong negative predictor of churn)
    log_odds -= tenure * 0.03

    # Internet service
    log_odds += np.where(internet_service == "Fiber optic", 0.8, 0)
    log_odds -= np.where(internet_service == "No", 0.5, 0)

    # No tech support or security
    log_odds += np.where(tech_support == "No", 0.4, 0)
    log_odds += np.where(online_security == "No", 0.4, 0)

    # Electronic check
    log_odds += np.where(payment_method == "Electronic check", 0.5, 0)

    # Senior citizen
    log_odds += np.where(senior_citizen == 1, 0.3, 0)

    # Monthly charges (normalized)
    log_odds += (monthly_charges - 50) * 0.008

    # No partner or dependents
    log_odds += np.where(partner == "No", 0.2, 0)
    log_odds += np.where(dependents == "No", 0.15, 0)

    # Convert log-odds to probability via sigmoid
    churn_prob = 1 / (1 + np.exp(-log_odds))
    churn_prob = np.clip(churn_prob, 0.02, 0.95)
    churn = np.array(["Yes" if rng.random() < p else "No" for p in churn_prob])

    df = pd.DataFrame({
        "customerID": customer_id,
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn,
    })

    return df


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    df = generate_churn_data()
    output_path = os.path.join(data_dir, "raw_data.csv")
    df.to_csv(output_path, index=False)

    churn_rate = (df["Churn"] == "Yes").mean()
    print(f"Generated {len(df)} customers → {output_path}")
    print(f"Churn rate: {churn_rate:.1%}")

"""
generate_sample_data.py — Creates a realistic synthetic Telco dataset
so you can run the pipeline without downloading the Kaggle CSV.

Usage:
    python generate_sample_data.py
Output:
    ../data/telco_churn.csv  (7043 rows, matching real Telco schema)
"""

import pandas as pd
import numpy as np
import os

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

N = 7043
os.makedirs("../data", exist_ok=True)


def generate():
    tenure = rng.integers(0, 73, N)

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        N, p=[0.55, 0.21, 0.24]
    )
    payment = rng.choice(
        ["Electronic check", "Mailed check",
         "Bank transfer (automatic)", "Credit card (automatic)"],
        N, p=[0.34, 0.23, 0.22, 0.21]
    )

    monthly = rng.uniform(18, 120, N).round(2)
    total = (monthly * tenure + rng.normal(0, 5, N)).clip(0).round(2)

    # Churn probability depends on contract, tenure, monthly charges
    contract_risk = np.where(contract == "Month-to-month", 0.35,
                    np.where(contract == "One year", 0.11, 0.03))
    tenure_risk = np.where(tenure < 6, 0.15, np.where(tenure < 24, 0.05, 0.0))
    churn_prob = (contract_risk + tenure_risk + (monthly / 120) * 0.1).clip(0, 1)
    churn = rng.binomial(1, churn_prob, N)

    yes_no = lambda p: rng.choice(["Yes", "No"], N, p=[p, 1 - p])

    df = pd.DataFrame({
        "customerID": [f"CUST-{i:05d}" for i in range(N)],
        "gender": rng.choice(["Male", "Female"], N),
        "SeniorCitizen": rng.choice([0, 1], N, p=[0.84, 0.16]),
        "Partner": yes_no(0.48),
        "Dependents": yes_no(0.30),
        "tenure": tenure,
        "PhoneService": yes_no(0.90),
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"], N, p=[0.42, 0.48, 0.10]),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], N, p=[0.34, 0.44, 0.22]),
        "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], N, p=[0.29, 0.50, 0.21]),
        "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], N, p=[0.34, 0.44, 0.22]),
        "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], N, p=[0.34, 0.44, 0.22]),
        "TechSupport": rng.choice(["Yes", "No", "No internet service"], N, p=[0.29, 0.49, 0.22]),
        "StreamingTV": rng.choice(["Yes", "No", "No internet service"], N, p=[0.38, 0.40, 0.22]),
        "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], N, p=[0.39, 0.39, 0.22]),
        "Contract": contract,
        "PaperlessBilling": yes_no(0.59),
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Churn": np.where(churn == 1, "Yes", "No"),
    })

    out = "../data/telco_churn.csv"
    df.to_csv(out, index=False)
    print(f"✅ Generated {N} rows → {out}")
    print(f"   Churn rate: {(churn == 1).mean():.2%}")


if __name__ == "__main__":
    generate()
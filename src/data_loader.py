"""
data_loader.py — Load and clean the Telco Customer Churn dataset.
"""

import pandas as pd
import numpy as np


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Load raw CSV, fix types, handle missing values, encode binary columns.

    Returns a clean DataFrame with:
      - TotalCharges as float
      - Churn as int (0/1)
      - SeniorCitizen confirmed as int
      - No null values
    """
    df = pd.read_csv(filepath)

    # TotalCharges is sometimes loaded as object due to whitespace
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing TotalCharges (new customers with 0 tenure)
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])

    # Binary target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Standardise Yes/No columns to 1/0
    binary_cols = [
        "Partner", "Dependents", "PhoneService",
        "PaperlessBilling", "MultipleLines"
    ]
    for col in binary_cols:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].map({"Yes": 1, "No": 0, "No phone service": 0})

    # Drop duplicates
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df
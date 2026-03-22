import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
df = pd.read_csv("credit_score.csv")

# Cleaning
df = df.replace("_", pd.NA)
df = df.replace("", pd.NA)

numeric_cols = [
    "age", "annual_income", "num_bank_accounts", "num_credit_card",
    "interest_rate", "num_of_loan", "delay_from_due_date",
    "outstanding_debt", "credit_utilization_ratio"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna()

# Features & target
X = df[numeric_cols]
y = df["credit_score"]

# Encode
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Model
model = LogisticRegression(max_iter=3000, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# Save artifacts
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(le, "models/encoder.pkl")

print("✅ Model training complete and saved!")
from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import pandas as pd
import numpy as np
import os
import secrets

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from model_utils import load_model

# =============================
# App Config
# =============================
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# =============================
# Load Model (NO TRAINING HERE)
# =============================
model, scaler, encoder = load_model()

# =============================
# Load Dataset (for dashboard only)
# =============================
df = pd.read_csv("credit_score.csv")

# =============================
# Static Charts Generation (run once)
# =============================
os.makedirs("static", exist_ok=True)

def generate_static_charts():
    if os.path.exists("static/chart1.png"):
        return

    # Map labels
    mapping = {'0': 'Good', '1': 'Standard', '2': 'Poor'}
    df['credit_score_label'] = df['credit_score'].astype(str).map(mapping)

    # 1. Distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x=df["credit_score_label"],
                  order=['Good', 'Standard', 'Poor'],
                  palette=['#2ecc71', '#f39c12', '#e74c3c'])
    plt.title("Credit Score Distribution")
    plt.savefig("static/chart1.png")
    plt.close()

    # 2. Income boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df["credit_score_label"],
                y=df["annual_income"],
                order=['Good', 'Standard', 'Poor'],
                palette=['#2ecc71', '#f39c12', '#e74c3c'])
    plt.title("Income vs Credit Score")
    plt.savefig("static/chart3.png")
    plt.close()

    # 3. Loans
    plt.figure(figsize=(10, 6))
    plt.hist(df["num_of_loan"], bins=10, color="#3498db")
    plt.title("Loan Distribution")
    plt.savefig("static/chart4.png")
    plt.close()

    # 4. Debt
    plt.figure(figsize=(10, 6))
    plt.hist(df["outstanding_debt"], bins=10, color="#e67e22")
    plt.title("Debt Distribution")
    plt.savefig("static/chart5.png")
    plt.close()

generate_static_charts()

# =============================
# HOME (Prediction Page)
# =============================
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    chart_exists = False
    form_data = {}

    if request.method == "POST":
        form_data = request.form.to_dict()

        try:
            # Get Inputs
            age = int(request.form.get('age', 0))
            income = float(request.form.get('income', 0))
            bank = int(request.form.get('bank', 0))
            card = int(request.form.get('card', 0))
            interest = float(request.form.get('interest', 0))
            loan = int(request.form.get('loan', 0))
            delay = int(request.form.get('delay', 0))
            debt = float(request.form.get('debt', 0))
            util = float(request.form.get('util', 0))

            # Feature Order MUST MATCH training
            features = np.array([[ 
                age, income, bank, card, interest,
                loan, delay, debt, util
            ]])

            features_scaled = scaler.transform(features)

            pred_encoded = model.predict(features_scaled)[0]
            pred_proba = model.predict_proba(features_scaled)[0]

            pred_class = encoder.inverse_transform([pred_encoded])[0]

            prob = max(pred_proba) * 100

            # Label Styling
            if pred_class == "0":
                label = "Good"
                color = "#2ecc71"
            elif pred_class == "1":
                label = "Standard"
                color = "#f39c12"
            else:
                label = "Poor"
                color = "#e74c3c"

            prediction = f"{label} ({prob:.1f}%)"

            # =============================
            # Dynamic Chart
            # =============================
            classes = ['Good', 'Standard', 'Poor']
            probs = pred_proba * 100

            plt.figure(figsize=(10, 6))
            bars = plt.bar(classes, probs,
                           color=['#2ecc71', '#f39c12', '#e74c3c'])

            bars[np.argmax(probs)].set_edgecolor("black")
            bars[np.argmax(probs)].set_linewidth(3)

            for i, v in enumerate(probs):
                plt.text(i, v + 1, f"{v:.1f}%", ha='center')

            plt.title("Prediction Probability", color=color)
            plt.savefig("static/dynamic_chart.png")
            plt.close()

            chart_exists = True

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template(
        "index.html",
        prediction=prediction,
        chart_exists=chart_exists,
        form_data=form_data
    )

# =============================
# DASHBOARD
# =============================
@app.route("/dashboard")
def dashboard():
    mapping = {'0': 'Good', '1': 'Standard', '2': 'Poor'}
    df['label'] = df['credit_score'].astype(str).map(mapping)

    good = len(df[df["label"] == "Good"])
    standard = len(df[df["label"] == "Standard"])
    poor = len(df[df["label"] == "Poor"])

    avg_income = int(df["annual_income"].mean())
    avg_debt = int(df["outstanding_debt"].mean())
    avg_util = round(df["credit_utilization_ratio"].mean(), 2)

    return render_template(
        "dashboard.html",
        total=len(df),
        good=good,
        standard=standard,
        poor=poor,
        avg_income=avg_income,
        avg_debt=avg_debt,
        avg_util=avg_util
    )

# =============================
# API ENDPOINT
# =============================
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.json

    features = np.array([[ 
        data["age"],
        data["income"],
        data["bank"],
        data["card"],
        data["interest"],
        data["loan"],
        data["delay"],
        data["debt"],
        data["util"]
    ]])

    features_scaled = scaler.transform(features)

    pred = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled).max()

    label = encoder.inverse_transform([pred])[0]

    return jsonify({
        "prediction": label,
        "confidence": round(prob * 100, 2)
    })

# =============================
# CLEAR SESSION
# =============================
@app.route("/clear")
def clear():
    session.clear()
    return redirect(url_for("home"))

# =============================
# RUN APP
# =============================
if __name__ == "__main__":
    app.run(debug=True)
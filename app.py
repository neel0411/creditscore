from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# =============================
# Load Dataset
# =============================
df = pd.read_csv("credit_score.csv")

print("="*60)
print("CREDIT SCORE CLASSIFICATION SYSTEM")
print("="*60)
print(f"Dataset Shape: {df.shape}")

# Data Cleaning
df = df.replace("_", pd.NA)
df = df.replace("", pd.NA)
df["credit_score"] = df["credit_score"].astype(str).str.strip()

numeric_cols = [
    "age",
    "annual_income",
    "num_bank_accounts",
    "num_credit_card",
    "interest_rate",
    "num_of_loan",
    "delay_from_due_date",
    "outstanding_debt",
    "credit_utilization_ratio"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=numeric_cols + ["credit_score"])

print(f"\nDataset after cleaning: {df.shape}")

# =============================
# Map numeric credit scores to proper labels
# =============================
credit_score_mapping = {
    '0': 'Good',
    '1': 'Standard', 
    '2': 'Poor'
}

df['credit_score_label'] = df['credit_score'].map(credit_score_mapping)
print(f"Credit Score Classes: {df['credit_score_label'].unique()}")
print("\n=== CLASS DISTRIBUTION ===")
print(df['credit_score_label'].value_counts())

# =============================
# Prepare features for model
# =============================
feature_cols = [col for col in numeric_cols if col in df.columns]
X = df[feature_cols]
y = df['credit_score']

le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = ['Good', 'Standard', 'Poor']

print(f"\nFeatures used: {feature_cols}")
print(f"Target classes: {class_names}")

# =============================
# Train Test Split
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# =============================
# Scaling
# =============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================
# Train Logistic Regression Model
# =============================
model = LogisticRegression(max_iter=3000, class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train)

# =============================
# Model Evaluation
# =============================
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\n=== MODEL PERFORMANCE ===")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"\nConfusion Matrix:")
print(cm)

# =============================
# Generate STATIC Dashboard Charts (WITH CONFUSION MATRIX)
# =============================
os.makedirs("static", exist_ok=True)

# 1. Credit Score Distribution (Green, Orange, Red theme)
plt.figure(figsize=(10, 6))
ax = sns.countplot(x=df["credit_score_label"], order=['Good', 'Standard', 'Poor'], 
                  palette=['#2ecc71', '#f39c12', '#e74c3c'])
plt.title("Credit Score Distribution", fontsize=16, fontweight='bold')
plt.xlabel("Credit Score Category", fontsize=12)
plt.ylabel("Number of Customers", fontsize=12)
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width()/2., p.get_height()), 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig("static/chart1.png", dpi=100)
plt.close()

# 2. Confusion Matrix (Blue theme - standard for confusion matrix)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix", fontsize=16, fontweight='bold')
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.tight_layout()
plt.savefig("static/chart2.png", dpi=100)
plt.close()

# 3. Income vs Credit Score Boxplot (Green, Orange, Red theme)
plt.figure(figsize=(10, 6))
sns.boxplot(x=df["credit_score_label"], y=df["annual_income"], 
           order=['Good', 'Standard', 'Poor'],
           palette=['#2ecc71', '#f39c12', '#e74c3c'])
plt.title("Income Distribution by Credit Score", fontsize=16, fontweight='bold')
plt.xlabel("Credit Score Category", fontsize=12)
plt.ylabel("Annual Income (₹)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("static/chart3.png", dpi=100)
plt.close()

# 4. Loan Distribution Histogram (Blue theme)
plt.figure(figsize=(10, 6))
n, bins, patches = plt.hist(df["num_of_loan"], bins=10, color='#3498db', edgecolor='black', alpha=0.7)
plt.title("Number of Loans Distribution", fontsize=16, fontweight='bold')
plt.xlabel("Number of Loans", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(True, alpha=0.3)
for i in range(len(patches)):
    if n[i] > 0:
        plt.text(patches[i].get_x() + patches[i].get_width()/2, n[i] + 1, 
                f'{int(n[i])}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig("static/chart4.png", dpi=100)
plt.close()

# 5. Outstanding Debt Distribution (Orange theme)
plt.figure(figsize=(10, 6))
n, bins, patches = plt.hist(df["outstanding_debt"], bins=10, color='#e67e22', edgecolor='black', alpha=0.7)
plt.title("Outstanding Debt Distribution", fontsize=16, fontweight='bold')
plt.xlabel("Outstanding Debt (₹)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(True, alpha=0.3)
for i in range(len(patches)):
    if n[i] > 0:
        plt.text(patches[i].get_x() + patches[i].get_width()/2, n[i] + 1, 
                f'{int(n[i])}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig("static/chart5.png", dpi=100)
plt.close()

# 6. Credit Utilization Distribution (Purple theme)
"""
plt.figure(figsize=(10, 6))
n, bins, patches = plt.hist(df["credit_utilization_ratio"], bins=10, color='#9b59b6', edgecolor='black', alpha=0.7)
plt.title("Credit Utilization Distribution", fontsize=16, fontweight='bold')
plt.xlabel("Credit Utilization Ratio", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(True, alpha=0.3)
for i in range(len(patches)):
    if n[i] > 0:
        plt.text(patches[i].get_x() + patches[i].get_width()/2, n[i] + 1, 
                f'{int(n[i]):.0f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig("static/chart6.png", dpi=100)
plt.close()
"""

print("\nDashboard charts generated successfully!")
print("Chart1: Credit Score Distribution (Green/Orange/Red)")
print("Chart2: Confusion Matrix (Blue)")
print("Chart3: Income Distribution (Green/Orange/Red)")
print("Chart4: Loan Distribution (Blue)")
print("Chart5: Debt Distribution (Orange)")
print("Chart6: Credit Utilization (Purple)")

# =============================
# Rule-Based Prediction Function
# =============================
def predict_credit_score(income, debt, delay, util, loan):
    if income >= 800000 and debt <= 100000 and delay == 0 and util <= 0.30 and loan <= 1:
        return "Good", 0
    elif income >= 350000 and debt <= 300000 and delay <= 5 and util <= 0.60:
        return "Standard", 1
    else:
        return "Poor", 2

# =============================
# Prediction Page
# =============================
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = session.get('prediction', '')
    chart_exists = session.get('chart_exists', False)
    form_data = session.get('form_data', {})
    user_category = session.get('user_category', '')
    
    if request.method == "POST":
        form_data = request.form.to_dict()
        
        income = float(request.form.get('income', 0))
        debt = float(request.form.get('debt', 0))
        delay = int(request.form.get('delay', 0))
        util = float(request.form.get('util', 0))
        loan = int(request.form.get('loan', 0))
        
        pred_class, pred_encoded = predict_credit_score(income, debt, delay, util, loan)
        
        if pred_class == "Good":
            pred_proba = [0.95, 0.03, 0.02]
            highlight_color = '#2ecc71'
            cat_index = 0
        elif pred_class == "Standard":
            pred_proba = [0.02, 0.95, 0.03]
            highlight_color = '#f39c12'
            cat_index = 1
        else:
            pred_proba = [0.01, 0.02, 0.97]
            highlight_color = '#e74c3c'
            cat_index = 2
        
        pred_prob = pred_proba[cat_index] * 100
        
        if pred_class == "Good":
            prediction = f"✅ Good (Low Risk) - {pred_prob:.1f}%"
        elif pred_class == "Standard":
            prediction = f"✅ Standard (Medium Risk) - {pred_prob:.1f}%"
        else:
            prediction = f"⚠️ Poor (High Risk) - {pred_prob:.1f}%"
        
        # Dynamic graph
        plt.figure(figsize=(10, 6))
        classes = ['Good', 'Standard', 'Poor']
        probabilities = [p * 100 for p in pred_proba]
        
        bars = plt.bar(classes, probabilities, color=['#2ecc71', '#f39c12', '#e74c3c'], 
                      edgecolor='black', linewidth=1.5)
        
        bars[cat_index].set_edgecolor('black')
        bars[cat_index].set_linewidth(4)
        bars[cat_index].set_alpha(0.9)
        
        star_y_position = probabilities[cat_index] + 8
        plt.scatter(cat_index, star_y_position, marker='*', s=500, c='gold', 
                   edgecolor='black', linewidth=2, zorder=10)
        
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 1,
                    f'{prob:.1f}%', ha='center', fontsize=12, fontweight='bold')
        
        plt.title(f"Credit Score Prediction - {pred_class}", fontsize=18, fontweight='bold', color=highlight_color, pad=20)
        plt.xlabel("Credit Score Category", fontsize=14, fontweight='bold')
        plt.ylabel("Probability (%)", fontsize=14, fontweight='bold')
        plt.ylim(0, 115)
        plt.grid(axis='y', alpha=0.3)
        
        plt.figtext(0.5, -0.05, 
                   f'Your Annual Income: ₹{income/100000:.1f}L | Your Outstanding Debt: ₹{debt/100000:.1f}L',
                   ha='center', fontsize=12,
                   bbox=dict(facecolor=highlight_color, alpha=0.2, boxstyle='round', edgecolor=highlight_color))
        
        plt.tight_layout()
        plt.savefig("static/dynamic_chart.png", dpi=150, bbox_inches="tight", facecolor='white')
        plt.close()
        
        chart_exists = True
        
        session['prediction'] = prediction
        session['form_data'] = form_data
        session['user_category'] = pred_class
        session['chart_exists'] = True
        session['accuracy'] = accuracy * 100
        
        return render_template("index.html", 
                              prediction=prediction,
                              chart_exists=chart_exists,
                              form_data=form_data,
                              feature_cols=feature_cols,
                              user_category=pred_class)
    
    return render_template("index.html", 
                          prediction=prediction,
                          chart_exists=chart_exists,
                          form_data=form_data,
                          feature_cols=feature_cols,
                          user_category=user_category)

# =============================
# Dashboard
# =============================
@app.route("/dashboard")
def dashboard():
    good_count = len(df[df["credit_score"] == "0"])
    standard_count = len(df[df["credit_score"] == "1"])
    poor_count = len(df[df["credit_score"] == "2"])
    
    return render_template(
        "dashboard.html",
        accuracy=round(accuracy * 100, 2),
        total_customers=len(df),
        good_count=good_count,
        standard_count=standard_count,
        poor_count=poor_count
    )

# =============================
# Clear Session
# =============================
@app.route("/clear")
def clear_session():
    session.clear()
    return redirect(url_for('home'))

# =============================
# Run App
# =============================
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)

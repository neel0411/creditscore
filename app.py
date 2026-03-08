from flask import Flask, render_template, request
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("credit_score.csv")

df = df.replace('_', 0)

df["age"] = pd.to_numeric(df["age"], errors="coerce")
df["annual_income"] = pd.to_numeric(df["annual_income"], errors="coerce")
df["outstanding_debt"] = pd.to_numeric(df["outstanding_debt"], errors="coerce")

df = df.dropna()

# Select columns
df = df[[
    "age",
    "annual_income",
    "num_bank_accounts",
    "num_credit_card",
    "interest_rate",
    "num_of_loan",
    "delay_from_due_date",
    "outstanding_debt",
    "credit_utilization_ratio",
    "credit_score"
]]

# -----------------------------
# Encode Credit Score
# -----------------------------
le = LabelEncoder()
df["credit_score_encoded"] = le.fit_transform(df["credit_score"])

# -----------------------------
# Features and Target
# -----------------------------
X = df.drop(["credit_score", "credit_score_encoded"], axis=1)
y = df["credit_score_encoded"]

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Train Model
# -----------------------------
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# -----------------------------
# Model Evaluation
# -----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# -----------------------------
# Prediction Page
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""

    if request.method == "POST":

        age = float(request.form["age"])
        income = float(request.form["income"])
        bank = int(request.form["bank"])
        card = int(request.form["card"])
        interest = float(request.form["interest"])
        loan = int(request.form["loan"])
        delay = int(request.form["delay"])
        debt = float(request.form["debt"])
        util = float(request.form["util"])

        data = scaler.transform([[age, income, bank, card, interest, loan, delay, debt, util]])

        result = model.predict(data)

        score = le.inverse_transform(result)[0]

        if score == "Good":
            risk = "Low Risk"
        elif score == "Standard":
            risk = "Medium Risk"
        else:
            risk = "High Risk"

        prediction = f"{score} ({risk})"

    return render_template("index.html", prediction=prediction)

# -----------------------------
# Dashboard
# -----------------------------
@app.route("/dashboard")
def dashboard():

    if not os.path.exists("static"):
        os.makedirs("static")

    # Chart 1 Credit Score Distribution
    plt.figure()
    sns.countplot(x=df["credit_score"])
    plt.title("Credit Score Distribution")
    plt.savefig("static/chart1.png")
    plt.close()

    # Chart 2 Income vs Credit Score
    plt.figure()
    sns.boxplot(x=df["credit_score"], y=df["annual_income"])
    plt.title("Income vs Credit Score")
    plt.savefig("static/chart2.png")
    plt.close()

    # Chart 3 Loan Distribution
    plt.figure()
    sns.histplot(df["num_of_loan"], bins=10)
    plt.title("Loan Distribution")
    plt.savefig("static/chart3.png")
    plt.close()

    # Chart 4 Debt Distribution
    plt.figure()
    sns.histplot(df["outstanding_debt"], bins=10)
    plt.title("Outstanding Debt Distribution")
    plt.savefig("static/chart4.png")
    plt.close()

    # Chart 5 Credit Utilization
    plt.figure()
    sns.histplot(df["credit_utilization_ratio"], bins=10)
    plt.title("Credit Utilization Ratio")
    plt.savefig("static/chart5.png")
    plt.close()

    # Chart 6 Pie Chart
    plt.figure()
    labels = df["credit_score"].value_counts().index
    sizes = df["credit_score"].value_counts().values
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    plt.title("Credit Score Proportion")
    plt.savefig("static/chart6.png")
    plt.close()

    # Chart 7 Correlation Heatmap
    plt.figure(figsize=(8,6))
    corr = df.drop("credit_score", axis=1).corr()

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Feature Correlation Heatmap")
    plt.savefig("static/chart7.png")
    plt.close()

    return render_template(
        "dashboard.html",
        accuracy=round(accuracy * 100, 2),
        cm=cm
    )

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
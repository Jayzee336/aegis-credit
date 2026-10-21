import json
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "trained_models"
MODEL_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(42)
models = {}
metrics = {}


# 1. LINEAR REGRESSION: Estimate loan amounts
n = 1000
income = rng.uniform(18000, 150000, n)
debt = rng.uniform(0, 50000, n)
credit_score = rng.integers(300, 851, n)
history = rng.uniform(0.5, 25, n)

X = np.column_stack([income, debt, credit_score, history])
y = (
    income * 0.30
    - debt * 0.40
    + credit_score * 35
    + history * 500
    + rng.normal(0, 3000, n)
)
y = np.clip(y, 1000, 100000)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
models["linear_regression"] = model
metrics["linear_regression"] = {
    "mae": round(float(mean_absolute_error(y_test, model.predict(X_test))), 2)
}


# Shared synthetic applicant data for classification models
X_app, y_risk = make_classification(
    n_samples=1200,
    n_features=4,
    n_informative=3,
    n_redundant=0,
    n_classes=2,
    class_sep=1.0,
    random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(
    X_app, y_risk, test_size=0.2, random_state=42
)

# 2. NAIVE BAYES: Classify messages
messages = [
    "Your loan application has been received",
    "Thank you for contacting customer support",
    "Your monthly payment is due",
    "Your account statement is ready",
    "Please verify your recent transaction",
    "We received your support request",
    "Congratulations your application is approved",
    "Your bank transfer was completed",
    "Urgent click this link to claim your prize",
    "Send your password to unlock your account",
    "You won a reward claim it immediately",
    "Verify your account at this suspicious link",
    "Your account will close unless you pay now",
    "Click here to receive free money",
    "Send your card details to claim your reward",
    "Your password has expired click this link",
]
message_labels = [
    0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1,
]
message_train, message_test, label_train, label_test = train_test_split(
    messages, message_labels, test_size=0.25, random_state=42, stratify=message_labels
)
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(message_train, label_train)
models["naive_bayes"] = model
metrics["naive_bayes"] = {
    "accuracy": round(float(accuracy_score(label_test, model.predict(message_test))), 3)
}


# 3. KNN: Find similar applicant profiles
model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
model.fit(X_train, y_train)
models["knn"] = model
metrics["knn"] = {
    "accuracy": round(float(accuracy_score(y_test, model.predict(X_test))), 3)
}


# 4. SVM: Classify credit risk
model = make_pipeline(StandardScaler(), SVC(probability=True, random_state=42))
model.fit(X_train, y_train)
models["svm"] = model
metrics["svm"] = {
    "accuracy": round(float(accuracy_score(y_test, model.predict(X_test))), 3)
}


# 5. DECISION TREE: Assist with loan decisions
X_decision, y_decision = make_classification(
    n_samples=1200,
    n_features=5,
    n_informative=4,
    n_redundant=0,
    n_classes=2,
    random_state=21,
)
X_train, X_test, y_train, y_test = train_test_split(
    X_decision, y_decision, test_size=0.2, random_state=42
)
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)
models["decision_tree"] = model
metrics["decision_tree"] = {
    "accuracy": round(float(accuracy_score(y_test, model.predict(X_test))), 3)
}


# 6. ARTIFICIAL NEURAL NETWORK: Detect suspicious transactions
X_fraud, y_fraud = make_classification(
    n_samples=1500,
    n_features=5,
    n_informative=4,
    n_redundant=0,
    weights=[0.88, 0.12],
    flip_y=0.02,
    random_state=17,
)
X_train, X_test, y_train, y_test = train_test_split(
    X_fraud, y_fraud, test_size=0.2, random_state=42, stratify=y_fraud
)
model = make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=600, random_state=42),
)
model.fit(X_train, y_train)
models["ann"] = model
metrics["ann"] = {
    "accuracy": round(float(accuracy_score(y_test, model.predict(X_test))), 3)
}


# Save trained models and evaluation metrics
joblib.dump(models, MODEL_DIR / "models.joblib")

with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

print("\nAegis Credit: Model training completed!")
print(f"Saved models to: {MODEL_DIR}")
print("\nEvaluation results:")
for name, result in metrics.items():
    print(f"{name}: {result}")

print(
    "\nNote: These models use synthetic demonstration data. "
    "Their results do not establish real-world banking performance."
)
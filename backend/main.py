import json
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "trained_models"

# Load the six trained models and their evaluation metrics.
models = joblib.load(MODEL_DIR / "models.joblib")

with open(MODEL_DIR / "metrics.json", "r", encoding="utf-8") as file:
    metrics = json.load(file)


app = FastAPI(
    title="Aegis Credit API",
    description=(
        "Educational demonstrations of six machine-learning algorithms "
        "using synthetic data. Not for real financial decisions."
    ),
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request schemas

class LoanEstimateInput(BaseModel):
    income: float = Field(gt=0, le=10_000_000)
    debt: float = Field(ge=0, le=10_000_000)
    credit_score: int = Field(ge=300, le=850)
    history: float = Field(gt=0, le=100)


class MessageInput(BaseModel):
    message: str = Field(min_length=1, max_length=5000)


class FourFeatureInput(BaseModel):
    feature_1: float
    feature_2: float
    feature_3: float
    feature_4: float


class FiveFeatureInput(BaseModel):
    feature_1: float
    feature_2: float
    feature_3: float
    feature_4: float
    feature_5: float


def get_class_result(model_name, features):
    """Return a prediction from a synthetic classification model."""
    model = models[model_name]
    data = np.array([features], dtype=float)
    prediction = int(model.predict(data)[0])

    result = {
        "model": model_name,
        "predicted_class": prediction,
        "class_label": f"Synthetic class {prediction}",
        "note": (
            "This is a demonstration using synthetic data, "
            "not a validated real-world financial assessment."
        ),
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(data)[0]
        result["class_probabilities"] = {
            str(label): round(float(probability), 4)
            for label, probability in zip(model.classes_, probabilities)
        }

    return result


# General endpoints

@app.get("/")
def home():
    return {
        "message": "Welcome to Aegis Credit API!",
        "status": "running",
        "models": len(models),
        "documentation": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/models")
def list_models():
    descriptions = {
        "linear_regression": {
            "name": "Linear Regression",
            "purpose": "Estimate a demonstration loan amount",
            "endpoint": "/predict/loan-estimate",
        },
        "naive_bayes": {
            "name": "Naive Bayes",
            "purpose": "Classify a message as ordinary or suspicious",
            "endpoint": "/predict/message",
        },
        "knn": {
            "name": "K-Nearest Neighbors",
            "purpose": "Classify a synthetic four-feature profile",
            "endpoint": "/predict/knn",
        },
        "svm": {
            "name": "Support Vector Machine",
            "purpose": "Classify a synthetic four-feature profile",
            "endpoint": "/predict/credit-risk",
        },
        "decision_tree": {
            "name": "Decision Tree",
            "purpose": "Classify a synthetic five-feature example",
            "endpoint": "/predict/loan-decision",
        },
        "ann": {
            "name": "Artificial Neural Network",
            "purpose": "Classify a synthetic five-feature transaction example",
            "endpoint": "/predict/fraud",
        },
    }

    return {
        "models": [
            {
                **details,
                "evaluation": metrics.get(model_id, {}),
            }
            for model_id, details in descriptions.items()
        ],
        "data_notice": (
            "All models were trained on synthetic demonstration data. "
            "Metrics do not establish real-world performance."
        ),
    }


# 1. Linear Regression: loan amount estimation

@app.post("/predict/loan-estimate")
def estimate_loan(data: LoanEstimateInput):
    features = np.array([[
        data.income,
        data.debt,
        data.credit_score,
        data.history,
    ]])

    prediction = float(
        models["linear_regression"].predict(features)[0]
    )

    return {
        "model": "Linear Regression",
        "estimated_amount": round(max(0.0, prediction), 2),
        "currency": "PHP",
        "note": (
            "Illustrative estimate from synthetic data, not a loan offer "
            "or a recommendation to lend."
        ),
    }


# 2. Naive Bayes: message classification

@app.post("/predict/message")
def classify_message(data: MessageInput):
    model = models["naive_bayes"]
    prediction = int(model.predict([data.message])[0])

    labels = {
        0: "Ordinary message",
        1: "Potentially suspicious message",
    }

    result = {
        "model": "Naive Bayes",
        "predicted_class": prediction,
        "classification": labels.get(prediction, "Unknown class"),
        "note": (
            "A demonstration only. Verify suspicious messages independently; "
            "this model may miss scams or flag legitimate messages."
        ),
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([data.message])[0]
        result["class_probabilities"] = {
            str(label): round(float(probability), 4)
            for label, probability in zip(model.classes_, probabilities)
        }

    return result


# 3. KNN: synthetic profile classification

@app.post("/predict/knn")
def classify_knn(data: FourFeatureInput):
    return get_class_result(
        "knn",
        [
            data.feature_1,
            data.feature_2,
            data.feature_3,
            data.feature_4,
        ],
    )


# 4. SVM: synthetic risk classification

@app.post("/predict/credit-risk")
def classify_credit_risk(data: FourFeatureInput):
    return get_class_result(
        "svm",
        [
            data.feature_1,
            data.feature_2,
            data.feature_3,
            data.feature_4,
        ],
    )


# 5. Decision Tree: synthetic decision classification

@app.post("/predict/loan-decision")
def classify_loan_example(data: FiveFeatureInput):
    return get_class_result(
        "decision_tree",
        [
            data.feature_1,
            data.feature_2,
            data.feature_3,
            data.feature_4,
            data.feature_5,
        ],
    )


# 6. Artificial Neural Network: synthetic transaction classification

@app.post("/predict/fraud")
def classify_transaction(data: FiveFeatureInput):
    return get_class_result(
        "ann",
        [
            data.feature_1,
            data.feature_2,
            data.feature_3,
            data.feature_4,
            data.feature_5,
        ],
    )
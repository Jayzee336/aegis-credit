# Aegis Credit

**An Educational FinTech Website Demonstrating Six Machine Learning Algorithms**

Aegis Credit is an educational FinTech project that demonstrates how machine learning can be applied to financial-related tasks, including loan estimation, message classification, credit risk analysis, loan decision support, and fraud detection.

The project combines a React-based frontend with a Python FastAPI backend. It is intended for academic demonstration and learning, not real financial decision-making.

## Features

- Interactive dashboard for exploring machine learning predictions.
- Loan amount estimation using Linear Regression.
- Message classification using Naive Bayes.
- Classification demonstrations using K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and Decision Tree.
- Artificial Neural Network (ANN) demonstration.
- Backend API with interactive Swagger documentation.
- Model evaluation metrics for educational comparison.

## Machine Learning Algorithms

| Algorithm | Educational application |
|---|---|
| Linear Regression | Estimates a numerical loan-related value. |
| Naive Bayes | Classifies messages into categories. |
| K-Nearest Neighbors (KNN) | Classifies samples based on nearby data points. |
| Support Vector Machine (SVM) | Separates samples into predicted classes. |
| Decision Tree | Makes predictions using a sequence of decision rules. |
| Artificial Neural Network (ANN) | Demonstrates learning patterns from input features. |

The project uses synthetic or demonstration data. Results are illustrative and should not be treated as evidence of real-world financial accuracy.

## Technology Stack

**Frontend**
- React
- Vite
- JavaScript
- CSS

**Backend**
- Python
- FastAPI
- Uvicorn
- scikit-learn
- pandas
- NumPy
- joblib

## Project Structure

```text
aegis-credit/
├── backend/
│   ├── main.py
│   ├── train_models.py
│   └── trained_models/
│       ├── models.joblib
│       └── metrics.json
├── frontend/
│   ├── src/
│   ├── package.json
│   └── package-lock.json
├── .gitignore
└── README.md
```

Some generated files and folders are intentionally excluded from Git.

## Requirements

Install the following before running the project:

- Git
- Python 3.13, or another Python version compatible with the project's dependencies
- Node.js and npm
- A code editor such as Visual Studio Code (optional)

## Installation and Setup

### 1. Clone the repository

Open a terminal and run:

```bash
git clone https://github.com/Jayzee336/aegis-credit.git
cd aegis-credit
```

### 2. Set up the backend

From the project root, run:

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install fastapi "uvicorn[standard]" scikit-learn pandas numpy joblib
```

If the trained model files are missing, generate them by running:

```bash
python train_models.py
```

Start the backend API:

```bash
python -m uvicorn main:app --reload
```

The API should be available at:

- API: http://127.0.0.1:8000
- Interactive API documentation: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

Keep this terminal open while using the website.

### 3. Set up the frontend

Open a **second terminal** and navigate to the project root, then run:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite, normally:

http://localhost:5173

Keep both the frontend and backend running while testing the application.

## Testing

1. Open the frontend dashboard.
2. Check that the dashboard loads correctly.
3. Open the API documentation at `http://127.0.0.1:8000/docs`.
4. Check the API health endpoint.
5. Try the available prediction demonstrations.
6. Review the returned predictions and model metrics.

## Troubleshooting

**Backend dependencies are missing**

Activate the backend virtual environment and reinstall the required packages.

**The frontend cannot connect to the backend**

Make sure the backend is running on port `8000` and the frontend is running on port `5173`.

**The frontend dependencies are missing**

Run `npm install` inside the `frontend` directory.

**Trained model files are missing**

Run `python train_models.py` inside the `backend` directory.

## Limitations and Disclaimer

Aegis Credit is an educational machine learning demonstration. Its models and datasets are not intended for actual loan approvals, credit assessments, fraud investigations, or other consequential financial decisions. Real-world deployment would require representative data, rigorous validation, security controls, fairness assessments, and appropriate human oversight.

## Project Purpose

This project demonstrates the integration of a modern web interface, a Python-based API, and multiple machine learning algorithms in a FinTech-themed application. It is designed to support learning, testing, and academic presentation.
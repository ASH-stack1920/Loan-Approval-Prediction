# 🏦 Loan Approval Prediction System

A complete, beginner-friendly **College Mini Project** that predicts whether a bank loan applicant is likely to be approved or rejected using Machine Learning (**Random Forest Classifier**) and provides an interactive web interface built with **Streamlit**.

---

## 📌 Project Overview

In the banking sector, evaluating loan applications manually is time-consuming and prone to human bias or error. This project automates the loan eligibility evaluation process based on historical loan application data from Kaggle.

### Key Highlights
- **End-to-End Pipeline**: Handles missing data, categorical feature encoding, and model inference in a unified Scikit-learn `Pipeline`.
- **Zero Data Leakage**: Preprocessing transformations are fit only on the training set and serialized alongside the model.
- **Accurate Model**: Uses a tuned `RandomForestClassifier` yielding **84.55% test accuracy**.
- **Interactive Web App**: Modern Streamlit UI for entering applicant details and receiving real-time prediction and probability score.
- **Beginner-Friendly**: Clean, modular, well-commented Python code without unnecessary complexity or external databases.

---

## 📂 Project Structure

```text
Loan-Approval-Prediction/
├── dataset/
│   ├── train.csv                 # Training dataset (with Loan_Status ground truth)
│   └── test.csv                  # Test dataset (unlabeled 367 applicants)
├── app.py                        # Streamlit web application
├── train_model.py                # Model training, evaluation & serialization script
├── predict_test.py               # Batch inference script for dataset/test.csv
├── loan_rf_pipeline.joblib       # Saved Random Forest model & preprocessing pipeline
├── predictions.csv               # Generated predictions for test.csv applicants
├── requirements.txt              # Project dependencies
├── .gitignore                    # Ignored files (cache, venv, etc.)
└── README.md                     # Project documentation
```

---

## 📊 Dataset Inspection & Features

The dataset comprises **614 records** and **12 input features** with `Loan_Status` as the target variable:

| Feature Name | Type | Description |
|---|---|---|
| `Gender` | Categorical | Male / Female |
| `Married` | Categorical | Applicant marital status (Yes / No) |
| `Dependents` | Categorical | Number of dependents (0, 1, 2, 3+) |
| `Education` | Categorical | Applicant education level (Graduate / Not Graduate) |
| `Self_Employed` | Categorical | Self-employment status (Yes / No) |
| `ApplicantIncome` | Numerical | Primary applicant monthly income (₹) |
| `CoapplicantIncome` | Numerical | Co-applicant monthly income (₹) |
| `LoanAmount` | Numerical | Loan amount requested (in thousands, e.g., 150 = ₹150,000) |
| `Loan_Amount_Term` | Numerical | Term of loan in months (e.g., 360 = 30 years) |
| `Credit_History` | Numerical/Binary | 1.0 = Guidelines met, 0.0 = Not met |
| `Property_Area` | Categorical | Urban, Semiurban, or Rural |
| **`Loan_Status`** | **Target** | **Y = Approved, N = Rejected** |

*(Note: `Loan_ID` is an identifier and is dropped before training).*

---

## ⚙️ Preprocessing & Modeling Pipeline

The Scikit-learn `Pipeline` and `ColumnTransformer` handle all data transformations automatically:

1. **Numerical Features** (`ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`):
   - Missing values imputed using the **Median** strategy (`SimpleImputer`).
2. **Categorical Features** (`Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `Property_Area`):
   - Missing values imputed using the **Most Frequent** strategy (`SimpleImputer`).
   - Encoded using **One-Hot Encoding** (`OneHotEncoder(handle_unknown='ignore')`).
3. **Classifier**:
   - `RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=5, random_state=42)`.
4. **Serialization**:
   - Entire pipeline (preprocessor + model) is saved to `loan_rf_pipeline.joblib`. When `app.py` receives user input, it feeds raw inputs directly into this pipeline.

---

## 📈 Model Performance & Evaluation

Trained on an 80/20 stratified split (491 training samples, 123 testing samples):

- **Test Accuracy**: **84.55%**

### Confusion Matrix
```text
                 Predicted 'Y' (Approved)   Predicted 'N' (Rejected)
Actual 'Y':                84                          1
Actual 'N':                18                         20
```

### Classification Report
```text
              precision    recall  f1-score   support

Rejected (N)       0.95      0.53      0.68        38
Approved (Y)       0.82      0.99      0.90        85

    accuracy                           0.85       123
   macro avg       0.89      0.76      0.79       123
weighted avg       0.86      0.85      0.83       123
```

---

## 🚀 How to Run the Project

### 1. Clone or Open the Project
Open the project directory in your terminal or VS Code:
```bash
cd Loan-Approval-Prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Train / Retrain the Model
To re-run training, evaluate metrics on the stratified test split of `train.csv`, and save `loan_rf_pipeline.joblib`:
```bash
python train_model.py
```
> **Note on Evaluation vs. Test Dataset**:
> `dataset/train.csv` contains ground-truth `Loan_Status` labels, so model evaluation (accuracy, confusion matrix, classification report) is performed on its 20% stratified test split. `dataset/test.csv` is an unlabeled dataset containing 367 applicants to evaluate in the real world.

### 4. Generate Predictions for `dataset/test.csv`
To batch-predict approval for all applicants in `dataset/test.csv` using the trained pipeline:
```bash
python predict_test.py
```
This script:
- Inspects `dataset/test.csv` (367 applicants) and validates features.
- Applies the saved preprocessor (median/most-frequent imputers, one-hot encoding).
- Generates predictions using the Random Forest classifier.
- Saves predictions with applicant identifiers (`Loan_ID`) to `predictions.csv`.

### 5. Launch the Streamlit Web Application
To test individual applicants interactively in the web UI:
```bash
streamlit run app.py
```
After running this command, open your browser at **http://localhost:8501**.

---

## 🖥️ Using the Web Application

1. Enter the applicant's profile (Gender, Marital Status, Dependents, Education, Self-Employment).
2. Enter the financial details (Applicant Income, Co-Applicant Income, Loan Amount, Loan Term, Credit History, Property Area).
3. Click **"🔍 Predict Loan Approval Status"**.
4. The application immediately displays:
   - **Decision Banner**: `🎉 LOAN APPROVED` or `❌ LOAN REJECTED`.
   - **Confidence Score**: Probability percentage calculated by the Random Forest model.
   - **Summary Card**: Review of all evaluated parameters.

---

## ☁️ Streamlit Community Cloud Deployment

When deploying to [Streamlit Community Cloud](https://share.streamlit.io), the cloud environment only has access to files that are committed and pushed to your **GitHub repository**.

### 1. Why the "Model Not Found" error occurred
If `loan_rf_pipeline.joblib` or `dataset/train.csv` only exist on your local machine and were never committed to Git, the cloud container cannot find them.

### 2. Built-in Auto-Training Feature
To ensure 100% reliable cloud deployments, `app.py` now includes an **automatic startup check**:
- If `loan_rf_pipeline.joblib` is present in the repository, it loads it immediately.
- If `loan_rf_pipeline.joblib` is absent, the app automatically trains the pipeline from `dataset/train.csv` in ~2 seconds upon startup and caches it in memory using `@st.cache_resource`.

### 3. Step-by-Step Deployment Checklist

1. **Commit and push required files to GitHub**:
   In your local project terminal, run:
   ```bash
   git add dataset/train.csv loan_rf_pipeline.joblib
   git commit -m "Add training dataset and model pipeline for cloud deployment"
   git push origin main
   ```
2. **Deploy on Streamlit Community Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io) and click **"New app"**.
   - Select your repository: `Loan-Approval-Prediction`.
   - Branch: `main`.
   - Main file path: `app.py`.
   - Click **"Deploy!"**.

### 4. What if your dataset cannot be made public?
- **Option A (Safest & Recommended)**: Commit and push only `loan_rf_pipeline.joblib` to GitHub. The serialized model contains mathematical weights and decision trees, but **no raw applicant data**.
- **Option B (Direct Upload)**: If neither file is in GitHub, the deployed app provides a secure built-in file uploader on screen where you can upload `train.csv` or `loan_rf_pipeline.joblib` directly in your browser.

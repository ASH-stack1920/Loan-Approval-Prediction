"""
Loan Approval Prediction - Model Training Script
College Mini Project using Random Forest Classifier
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def locate_dataset():
    """Locates the dataset file in the workspace."""
    candidate_paths = [
        os.path.join("dataset", "train.csv"),
        "train.csv",
        "train_u6lujuX_CVtuZ9i.csv",
        os.path.join("dataset", "train_u6lujuX_CVtuZ9i.csv")
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            print(f"[INFO] Found dataset at: {path}")
            return path
    raise FileNotFoundError(
        "Could not find dataset! Please place train.csv inside the 'dataset/' folder."
    )


def inspect_dataset(df):
    """Prints a clear, beginner-friendly inspection of the dataset."""
    print("=" * 60)
    print("               DATASET INSPECTION")
    print("=" * 60)
    print(f"Total Rows (Records):    {df.shape[0]}")
    print(f"Total Columns (Features): {df.shape[1]}")
    
    print("\nColumns & Data Types:")
    for col, dtype in df.dtypes.items():
        print(f" - {col:<22}: {dtype}")
        
    print("\nMissing Values Count:")
    missing = df.isnull().sum()
    for col, count in missing.items():
        if count > 0:
            pct = (count / len(df)) * 100
            print(f" - {col:<22}: {count} missing ({pct:.1f}%)")
    if missing.sum() == 0:
        print(" - No missing values found.")
        
    if "Loan_Status" in df.columns:
        print("\nTarget Column (Loan_Status) Distribution:")
        counts = df["Loan_Status"].value_counts(dropna=False)
        for label, cnt in counts.items():
            print(f" - {label}: {cnt} ({(cnt / len(df)) * 100:.1f}%)")
    print("=" * 60 + "\n")


def train_and_save_pipeline(dataset_path=None, model_filename="loan_rf_pipeline.joblib", verbose=True):
    """
    Trains the Random Forest preprocessing and classification pipeline on train.csv,
    evaluates it on an 80/20 stratified validation split, saves the artifact to disk,
    and returns the fitted pipeline.
    """
    if verbose:
        print("\n>>> Step 1: Loading Dataset...")
    if dataset_path is None:
        dataset_path = locate_dataset()
    df = pd.read_csv(dataset_path)

    if verbose:
        print("\n>>> Step 2: Inspecting Dataset...")
        inspect_dataset(df)

    # Validate target column
    target_col = "Loan_Status"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns: {list(df.columns)}")

    # Drop identifier column if present
    features_df = df.copy()
    if "Loan_ID" in features_df.columns:
        features_df = features_df.drop(columns=["Loan_ID"])
        if verbose:
            print("[INFO] Dropped 'Loan_ID' (not a predictive feature).")

    # Separate features and target
    X = features_df.drop(columns=[target_col])
    y = features_df[target_col].astype(str).str.strip().str.upper()

    # Identify numerical and categorical columns
    categorical_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
    numerical_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]

    # Filter to ensure only existing columns are included
    categorical_cols = [c for c in categorical_cols if c in X.columns]
    numerical_cols = [c for c in numerical_cols if c in X.columns]

    if verbose:
        print(f"[INFO] Categorical columns ({len(categorical_cols)}): {categorical_cols}")
        print(f"[INFO] Numerical columns ({len(numerical_cols)}): {numerical_cols}")
        print("\n>>> Step 3: Splitting Dataset (80% Train, 20% Test)...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    if verbose:
        print(f"[INFO] Training samples: {len(X_train)}")
        print(f"[INFO] Testing samples:  {len(X_test)}")
        print("\n>>> Step 4: Building Preprocessing Pipeline & Model...")

    # Numerical Preprocessing: Median Imputation
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    # Categorical Preprocessing: Most Frequent Imputation + One-Hot Encoding
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Combined Preprocessor using ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_pipeline, numerical_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    # Random Forest Classifier Pipeline
    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=5,
            random_state=42
        ))
    ])

    if verbose:
        print("\n>>> Step 5: Training Random Forest Classifier...")
    model_pipeline.fit(X_train, y_train)
    if verbose:
        print("[SUCCESS] Model training complete!")

    if verbose:
        print("\n>>> Step 6: Evaluating Model on Test Data...")
        y_pred = model_pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=["Y", "N"])

        print("=" * 60)
        print("               MODEL EVALUATION RESULTS")
        print("=" * 60)
        print(f"Test Accuracy: {acc * 100:.2f}%\n")
        print("Confusion Matrix (Labels: ['Approved (Y)', 'Rejected (N)']):")
        print(f"                 Predicted 'Y'   Predicted 'N'")
        print(f"  Actual 'Y':         {cm[0][0]:<14} {cm[0][1]}")
        print(f"  Actual 'N':         {cm[1][0]:<14} {cm[1][1]}")
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=["Rejected (N)", "Approved (Y)"]))
        print("=" * 60)

    if model_filename:
        if verbose:
            print("\n>>> Step 7: Saving Model & Pipeline using Joblib...")
        joblib.dump(model_pipeline, model_filename)
        if verbose:
            print(f"[SUCCESS] Saved complete pipeline and model to: {model_filename}")

    return model_pipeline


def main():
    model_pipeline = train_and_save_pipeline(verbose=True)

    # Verify saved model with a test prediction
    print("\n>>> Step 8: Verifying Saved Pipeline...")
    model_filename = "loan_rf_pipeline.joblib"
    loaded_pipeline = joblib.load(model_filename)
    sample_record = pd.DataFrame([{
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "0",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 5000.0,
        "CoapplicantIncome": 1500.0,
        "LoanAmount": 130.0,
        "Loan_Amount_Term": 360.0,
        "Credit_History": 1.0,
        "Property_Area": "Urban"
    }])
    sample_pred = loaded_pipeline.predict(sample_record)[0]
    sample_proba = loaded_pipeline.predict_proba(sample_record)[0]
    print(f"[INFO] Verification prediction on sample record: {sample_pred}")
    print(f"[INFO] Probabilities [N, Y]: {sample_proba}")
    print("[SUCCESS] Saved model verified successfully!\n")


if __name__ == "__main__":
    main()

"""
Loan Approval Prediction - Batch Prediction Script for Test Dataset
Loads the trained Random Forest pipeline and generates predictions for dataset/test.csv.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib


def locate_test_dataset():
    """
    Locates the test dataset in the workspace.
    Checks 'dataset/test.csv' first, followed by known alternative filenames.
    """
    candidate_paths = [
        os.path.join("dataset", "test.csv"),
        "test.csv",
        os.path.join("dataset", "test_Y3wMUE5_7gLdaTN.csv"),
        "test_Y3wMUE5_7gLdaTN.csv"
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            print(f"[INFO] Found test dataset at: {path}")
            return path
    raise FileNotFoundError(
        "Could not find test dataset! Please place 'test.csv' inside the 'dataset/' folder."
    )


def locate_model_pipeline():
    """
    Locates the serialized Random Forest pipeline file.
    """
    candidate_paths = [
        "loan_rf_pipeline.joblib",
        os.path.join("dataset", "loan_rf_pipeline.joblib")
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            print(f"[INFO] Found trained model pipeline at: {path}")
            return path
    raise FileNotFoundError(
        "Could not find 'loan_rf_pipeline.joblib'! Please run 'python train_model.py' first."
    )


def inspect_test_dataset(df):
    """
    Inspects test dataset columns, missing values, and checks for the Loan_Status target.
    """
    print("=" * 65)
    print("               TEST DATASET INSPECTION")
    print("=" * 65)
    print(f"Total Rows (Records to Predict): {df.shape[0]}")
    print(f"Total Columns:                   {df.shape[1]}")

    print("\nColumns & Data Types:")
    for col, dtype in df.dtypes.items():
        print(f" - {col:<22}: {dtype}")

    print("\nMissing Values Count:")
    missing = df.isnull().sum()
    has_missing = False
    for col, count in missing.items():
        if count > 0:
            pct = (count / len(df)) * 100
            print(f" - {col:<22}: {count} missing ({pct:.1f}%)")
            has_missing = True
    if not has_missing:
        print(" - No missing values found.")

    print("\nTarget Column Check:")
    if "Loan_Status" in df.columns:
        print(" - Target column 'Loan_Status' IS present in the test dataset.")
    else:
        print(" - Target column 'Loan_Status' is NOT present in test.csv.")
        print("   [NOTE] As per standard machine learning and Kaggle competition practices,")
        print("          test.csv is an unlabeled dataset for generating real-world predictions.")
        print("          Accuracy cannot be calculated on test.csv without ground-truth labels.")
        print("          Model performance (84.55% accuracy) is evaluated on the stratified")
        print("          validation split of train.csv in train_model.py.")
    print("=" * 65 + "\n")


def generate_and_save_predictions(output_csv="predictions.csv"):
    """
    Loads test data, applies trained pipeline, and saves predictions to CSV.
    """
    print(">>> Step 1: Locating Test Dataset...")
    test_path = locate_test_dataset()
    test_df = pd.read_csv(test_path)

    print("\n>>> Step 2: Inspecting Test Dataset...")
    inspect_test_dataset(test_df)

    print(">>> Step 3: Loading Pre-trained Model Pipeline...")
    model_path = locate_model_pipeline()
    pipeline = joblib.load(model_path)
    print(f"[SUCCESS] Pipeline loaded successfully.")
    print(f"[INFO] Pipeline model classes: {list(pipeline.classes_)}")

    print("\n>>> Step 4: Preparing Features for Prediction...")
    # Preserve applicant identifier if present
    id_col = None
    if "Loan_ID" in test_df.columns:
        id_col = "Loan_ID"
    else:
        # Check if another ID-like column exists
        for col in test_df.columns:
            if "id" in col.lower():
                id_col = col
                break

    # Exclude non-predictive ID column and target column (if any)
    cols_to_drop = []
    if id_col:
        cols_to_drop.append(id_col)
    if "Loan_Status" in test_df.columns:
        cols_to_drop.append("Loan_Status")

    X_test = test_df.drop(columns=cols_to_drop) if cols_to_drop else test_df.copy()
    print(f"[INFO] Features fed to model ({X_test.shape[1]}): {list(X_test.columns)}")

    print("\n>>> Step 5: Generating Predictions...")
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)

    # Find the index of class 'Y' for approval probability
    classes_list = list(pipeline.classes_)
    y_class_idx = classes_list.index("Y") if "Y" in classes_list else 1

    print("\n>>> Step 6: Building Predictions DataFrame...")
    if id_col:
        results_df = pd.DataFrame({
            id_col: test_df[id_col],
            "Loan_Status": predictions
        })
    else:
        results_df = pd.DataFrame({
            "Applicant_ID": range(1, len(predictions) + 1),
            "Loan_Status": predictions
        })

    # Save to output CSV file
    results_df.to_csv(output_csv, index=False)
    print(f"[SUCCESS] Saved {len(results_df)} predictions to '{output_csv}'!")

    # Summary Statistics
    total_count = len(results_df)
    approved_count = (results_df["Loan_Status"] == "Y").sum()
    rejected_count = (results_df["Loan_Status"] == "N").sum()
    approved_pct = (approved_count / total_count) * 100
    rejected_pct = (rejected_count / total_count) * 100

    print("\n" + "=" * 65)
    print("               PREDICTION SUMMARY")
    print("=" * 65)
    print(f"Total Test Applicants Processed: {total_count}")
    print(f"Predicted Approved (Y):          {approved_count} ({approved_pct:.1f}%)")
    print(f"Predicted Rejected (N):          {rejected_count} ({rejected_pct:.1f}%)")
    print("-" * 65)
    print("Sample Preview of predictions.csv (First 10 Rows):")
    print(results_df.head(10).to_string(index=False))
    print("-" * 65)
    print(f"File successfully written to: {os.path.abspath(output_csv)}")
    print("=" * 65 + "\n")

    return results_df


def main():
    print("\n" + "#" * 65)
    print("#  LOAN APPROVAL PREDICTION - TEST SET INFERENCE SCRIPT        #")
    print("#" * 65 + "\n")
    generate_and_save_predictions("predictions.csv")


if __name__ == "__main__":
    main()

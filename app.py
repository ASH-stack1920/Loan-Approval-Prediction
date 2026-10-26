"""
Loan Approval Prediction - Interactive Web App
Built with Streamlit and Scikit-Learn Random Forest
"""

import os
import streamlit as st
import pandas as pd
import joblib

# Set Page Config
st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Clean, Modern College Project UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F3F4F6;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 15px;
    }
    .result-box-approved {
        background-color: #ECFDF5;
        border: 2px solid #10B981;
        border-radius: 12px;
        padding: 22px;
        text-align: center;
        margin-top: 20px;
    }
    .result-box-rejected {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        border-radius: 12px;
        padding: 22px;
        text-align: center;
        margin-top: 20px;
    }
    .stButton>button {
        width: 100%;
        background-color: #2563EB;
        color: white;
        font-weight: 600;
        padding: 0.6rem 1rem;
        border-radius: 8px;
        border: none;
        transition: 0.2s;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_trained_pipeline():
    """Loads the serialized Random Forest pipeline from disk."""
    model_paths = [
        "loan_rf_pipeline.joblib",
        os.path.join("dataset", "loan_rf_pipeline.joblib")
    ]
    for p in model_paths:
        if os.path.exists(p):
            return joblib.load(p), p
    return None, None


# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=80)
    st.title("Project Overview")
    st.info(
        "**College Mini Project**\n\n"
        "**Topic:** Loan Approval Prediction\n\n"
        "**Algorithm:** Random Forest Classifier\n\n"
        "**Features:** Categorical Encoding, Missing Value Imputation, Scalable Scikit-Learn Pipeline"
    )
    
    st.markdown("---")
    st.subheader("💡 Key Guidelines")
    st.write("• **Credit History** is typically the strongest predictor in bank loan approval.")
    st.write("• **Coapplicant Income** can significantly boost approval chance for higher loan amounts.")
    st.write("• Loan Amount is entered in **thousands** (e.g., 150 = ₹150,000).")


# --- MAIN CONTENT ---
st.markdown('<div class="main-header">🏦 Loan Approval Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Enter the applicant\'s demographic, credit, and financial details to check loan eligibility in real-time.</div>', unsafe_allow_html=True)

pipeline, model_path = load_trained_pipeline()

if pipeline is None:
    st.error("⚠️ **Model Not Found!**")
    st.warning("Please train the model first by running `python train_model.py` in your terminal. This creates `loan_rf_pipeline.joblib`.")
    st.stop()

# --- INPUT FORM ---
with st.form("loan_prediction_form"):
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.subheader("👤 Applicant Profile")
        gender = st.selectbox("Gender", options=["Male", "Female"], index=0)
        married = st.selectbox("Marital Status", options=["No", "Yes"], index=1)
        dependents = st.selectbox("Number of Dependents", options=["0", "1", "2", "3+"], index=0)
        education = st.selectbox("Education Level", options=["Graduate", "Not Graduate"], index=0)
        self_employed = st.selectbox("Self Employed?", options=["No", "Yes"], index=0)
        property_area = st.selectbox("Property Area", options=["Urban", "Semiurban", "Rural"], index=1)

    with col2:
        st.subheader("💰 Financial & Loan Details")
        applicant_income = st.number_input(
            "Applicant Monthly Income (₹)",
            min_value=0,
            max_value=100000,
            value=5000,
            step=250,
            help="Primary applicant's gross monthly income in INR (₹)."
        )
        coapplicant_income = st.number_input(
            "Coapplicant Monthly Income (₹)",
            min_value=0,
            max_value=50000,
            value=1500,
            step=250,
            help="Co-signer or spouse's gross monthly income in INR (₹) (can be 0)."
        )
        loan_amount = st.number_input(
            "Loan Amount (₹ in Thousands)",
            min_value=1,
            max_value=1000,
            value=130,
            step=5,
            help="Requested loan amount in thousands. E.g., 150 = ₹150,000."
        )
        loan_term = st.selectbox(
            "Loan Term (Months)",
            options=[360.0, 180.0, 240.0, 300.0, 120.0, 84.0, 60.0, 36.0, 12.0],
            index=0,
            format_func=lambda x: f"{int(x)} months ({int(x/12)} years)"
        )
        credit_history = st.selectbox(
            "Credit History Guidelines Met?",
            options=[1.0, 0.0],
            index=0,
            format_func=lambda x: "Yes (Score 1.0 - Good credit standing)" if x == 1.0 else "No (Score 0.0 - Bad/no credit history)"
        )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.form_submit_button("🔍 Predict Loan Approval Status")


# --- PREDICTION LOGIC ---
if predict_button:
    # Build single-row DataFrame matching the exact training columns
    input_df = pd.DataFrame([{
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": float(applicant_income),
        "CoapplicantIncome": float(coapplicant_income),
        "LoanAmount": float(loan_amount),
        "Loan_Amount_Term": float(loan_term),
        "Credit_History": float(credit_history),
        "Property_Area": property_area
    }])

    # Run Prediction through the trained pipeline
    try:
        prediction = pipeline.predict(input_df)[0]
        probabilities = pipeline.predict_proba(input_df)[0]
        classes = list(pipeline.classes_)

        # Probability for 'Y' (Approved) and 'N' (Rejected)
        prob_y = probabilities[classes.index("Y")] if "Y" in classes else 0.5
        prob_n = probabilities[classes.index("N")] if "N" in classes else 0.5

        st.markdown("---")
        st.subheader("📋 Decision Result")

        if prediction == "Y":
            st.markdown(f"""
            <div class="result-box-approved">
                <h2 style="color: #065F46; margin: 0;">🎉 LOAN APPROVED</h2>
                <p style="font-size: 1.15rem; color: #047857; margin-top: 8px;">
                    Congratulations! The applicant meets the eligibility criteria for the requested loan.
                </p>
                <div style="font-size: 1.05rem; font-weight: 600; color: #065F46; margin-top: 10px;">
                    Approval Confidence: {prob_y * 100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.balloons()

        else:
            st.markdown(f"""
            <div class="result-box-rejected">
                <h2 style="color: #991B1B; margin: 0;">❌ LOAN REJECTED</h2>
                <p style="font-size: 1.15rem; color: #B91C1C; margin-top: 8px;">
                    The applicant does not currently meet the necessary criteria for approval.
                </p>
                <div style="font-size: 1.05rem; font-weight: 600; color: #991B1B; margin-top: 10px;">
                    Rejection Likelihood: {prob_n * 100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Summary of Applicant's Inputs
        with st.expander("🔎 View Details of Evaluated Applicant"):
            summary_col1, summary_col2 = st.columns(2)
            with summary_col1:
                st.write(f"• **Total Household Income:** ₹{applicant_income + coapplicant_income:,}/month")
                st.write(f"• **Requested Loan Amount:** ₹{int(loan_amount * 1000):,}")
                st.write(f"• **Loan Term:** {int(loan_term)} months ({int(loan_term / 12)} years)")
            with summary_col2:
                st.write(f"• **Credit History Met:** {'Yes' if credit_history == 1.0 else 'No'}")
                st.write(f"• **Property Area:** {property_area}")
                st.write(f"• **Education:** {education}")

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")

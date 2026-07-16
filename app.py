import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# LOAD TRAINED PIPELINE
# ==========================================================

@st.cache_resource
def load_model():
    return joblib.load("Model/loan_approval_pipeline.pkl")

model = load_model()

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

/* ==========================================================
MAIN BACKGROUND
========================================================== */

.main{
    background-color:#F8FAFC;
}

/* ==========================================================
PAGE TITLES
========================================================== */

h1{
    color:#0F172A;
    text-align:center;
    font-weight:700;
}

h2,h3{
    color:#1E3A8A;
}

/* ==========================================================
METRIC CARDS
========================================================== */

.metric-card{
    background:linear-gradient(135deg,#2563EB,#1D4ED8);
    color:white;
    padding:20px;
    border-radius:15px;
    text-align:center;
    box-shadow:0 5px 12px rgba(0,0,0,.18);
    margin-bottom:15px;
}

.metric-card h2,
.metric-card h3,
.metric-card p{
    color:white;
    margin:0;
}

/* ==========================================================
BUTTON
========================================================== */

.stButton > button{
    width:100%;
    height:55px;
    border:none;
    border-radius:12px;
    background:#2563EB;
    color:white;
    font-size:18px;
    font-weight:bold;
}

.stButton > button:hover{
    background:#1E40AF;
    color:white;
}

/* ==========================================================
SIDEBAR
========================================================== */

[data-testid="stSidebar"]{
    background:#2B2D38;
}

/* Sidebar headings */

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3{
    color:#FFFFFF !important;
}

/* Sidebar labels */

[data-testid="stSidebar"] label{
    color:#FFFFFF !important;
    font-weight:600;
}

/* Sidebar description */

[data-testid="stSidebar"] p{
    color:#C7CBD6 !important;
}

/* Selected values inside dropdowns */

[data-testid="stSidebar"] [data-baseweb="select"] *{
    color:#FFFFFF !important;
}

/* Number inputs */

[data-testid="stSidebar"] input{
    color:#FFFFFF !important;
}

/* ==========================================================
CONTAINERS
========================================================== */

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

/* ==========================================================
SUCCESS / ERROR
========================================================== */

.stSuccess,
.stError,
.stWarning{
    border-radius:12px;
}

/* ==========================================================
FOOTER
========================================================== */

.footer{
    text-align:center;
    color:#6B7280;
    font-size:14px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# TITLE
# ==========================================================

st.title("🏦 Loan Approval Prediction System")

st.markdown(
"""
Predict whether a loan application is likely to be **Approved**
or **Rejected** using a Machine Learning model trained on
historical loan application data.
"""
)

st.divider()

# ==========================================================
# SIDEBAR HEADER
# ==========================================================

st.sidebar.markdown(
    """
    <h1 style="
        color:white;
        font-size:32px;
        font-weight:700;
        margin-bottom:5px;
        text-align:left;">
        Loan Application
    </h1>

    <p style="
        color:#C7CBD6;
        font-size:16px;
        margin-top:0;
        margin-bottom:25px;">
        Enter your details below
    </p>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# PERSONAL INFORMATION
# ==========================================================

st.sidebar.subheader("👤 Personal Information")

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

married = st.sidebar.selectbox(
    "Marital Status",
    ["Yes", "No"]
)

dependents = st.sidebar.selectbox(
    "Dependents",
    ["0", "1", "2", "3+"]
)

education = st.sidebar.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

self_employed = st.sidebar.selectbox(
    "Self Employed",
    ["No", "Yes"]
)

st.sidebar.divider()

# ==========================================================
# FINANCIAL INFORMATION
# ==========================================================

st.sidebar.subheader("💰 Financial Information")

applicant_income = st.sidebar.number_input(
    "Applicant Income",
    min_value=0.0,
    value=5000.0,
    step=100.0
)

coapplicant_income = st.sidebar.number_input(
    "Co-applicant Income",
    min_value=0.0,
    value=0.0,
    step=100.0
)

loan_amount = st.sidebar.number_input(
    "Loan Amount (in thousands)",
    min_value=0.0,
    value=120.0,
    step=1.0
)

loan_term = st.sidebar.selectbox(
    "Loan Amount Term (Days)",
    [12, 36, 60, 84, 120, 180, 240, 300, 360, 480]
)

credit_history = st.sidebar.selectbox(
    "Credit History",
    [1, 0],
    format_func=lambda x: "Good" if x == 1 else "Poor"
)

st.sidebar.divider()

# ==========================================================
# PROPERTY INFORMATION
# ==========================================================

st.sidebar.subheader("🏠 Property Information")

property_area = st.sidebar.selectbox(
    "Property Area",
    ["Urban", "Semiurban", "Rural"]
)

st.sidebar.write("")

# ==========================================================
# PREDICT BUTTON
# ==========================================================

predict = st.sidebar.button(
    "🔍 Predict Loan Approval",
    use_container_width=True
)

# ==========================================================
# CREATE INPUT DATAFRAME
# ==========================================================

sample_data = pd.DataFrame({

    "Gender": [gender],
    "Married": [married],
    "Dependents": [dependents],
    "Education": [education],
    "Self_Employed": [self_employed],
    "ApplicantIncome": [applicant_income],
    "CoapplicantIncome": [coapplicant_income],
    "LoanAmount": [loan_amount],
    "Loan_Amount_Term": [loan_term],
    "Credit_History": [credit_history],
    "Property_Area": [property_area]

})

# ==========================================================
# MAKE PREDICTION
# ==========================================================

if predict:

    prediction = model.predict(sample_data)[0]

    confidence = None
    approval_probability = None
    rejection_probability = None

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(sample_data)[0]

        rejection_probability = probabilities[0] * 100
        approval_probability = probabilities[1] * 100

        confidence = max(probabilities) * 100

    total_income = applicant_income + coapplicant_income
    
# ==========================================================
# PREDICTION RESULTS
# ==========================================================

    st.divider()

    if prediction == 1:

        st.success("## ✅ Loan Approved")

        st.write(
            """
            Based on the information provided, the applicant is
            likely to qualify for the requested loan.
            """
        )

    else:

        st.error("## ❌ Loan Rejected")

        st.write(
            """
            Based on the information provided, the applicant is
            unlikely to qualify for the requested loan.
            """
        )

    # ======================================================
    # SUMMARY CARDS
    # ======================================================

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(f"""
        <div class="metric-card">
            <h3>💰 Total Income</h3>
            <h2>₦ {total_income:,.0f}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        loan_status = "Approved" if prediction == 1 else "Rejected"

        st.markdown(f"""
        <div class="metric-card">
            <h3>📄 Loan Status</h3>
            <h2>{loan_status}</h2>
        </div>
        """, unsafe_allow_html=True)

    # ======================================================
    # CONFIDENCE
    # ======================================================

    if confidence is not None:

        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Confidence Score</h3>
            <h2>{confidence:.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)

        st.progress(confidence/100)

    # ======================================================
    # PROBABILITIES
    # ======================================================

    if approval_probability is not None:

        st.subheader("Prediction Probability")

        prob_col1, prob_col2 = st.columns(2)

        with prob_col1:

            st.metric(
                "Approval Probability",
                f"{approval_probability:.2f}%"
            )

        with prob_col2:

            st.metric(
                "Rejection Probability",
                f"{rejection_probability:.2f}%"
            )

    st.divider()

    # ======================================================
    # RECOMMENDATION
    # ======================================================

    st.subheader("💡 Recommendation")

    if prediction == 1:

        st.success(
            """
            The applicant has a strong likelihood of loan approval
            based on the submitted financial information.
            """
        )

    else:

        st.warning(
            """
            Approval is unlikely. Improving credit history,
            increasing income, or requesting a smaller loan amount
            may improve future approval chances.
            """
        )
        
# ======================================================
# APPLICATION DETAILS
# ======================================================

st.divider()

st.subheader("📋 Submitted Application")

display_data = sample_data.copy()

display_data.columns = [
    "Gender",
    "Marital Status",
    "Dependents",
    "Education",
    "Self Employed",
    "Applicant Income",
    "Co-applicant Income",
    "Loan Amount",
    "Loan Term",
    "Credit History",
    "Property Area"
]

display_data.index = ["Applicant"]

st.dataframe(
    display_data,
    use_container_width=True
)

# ======================================================
# DOWNLOAD APPLICATION
# ======================================================

csv = display_data.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Application Details",
    data=csv,
    file_name="loan_application_details.csv",
    mime="text/csv",
    use_container_width=True
)
        
# ==========================================================
# DISCLAIMER
# ==========================================================

st.divider()

st.info(
    """
    **Disclaimer**

    This application is intended for educational and demonstration purposes only.
    Predictions are generated using a Machine Learning model trained on historical
    loan application data and should not be considered as official loan approval
    decisions. Financial institutions evaluate additional factors before making
    lending decisions.
    """
)

# ==========================================================
# FOOTER
# ==========================================================

st.markdown(
"""
<div class="footer">

<hr style="margin-top:30px;margin-bottom:20px;">

<p style="font-size:15px;">
🏦 <b>Loan Approval Prediction System</b>
</p>

<p style="font-size:13px;">
Developed with ❤️ using
Python • Scikit-Learn • Streamlit
</p>

</div>
""",
unsafe_allow_html=True
)        
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

/* ===============================
Main Background
=============================== */

.main{
    background-color:#F8FAFC;
}

/* ===============================
Page Title
=============================== */

h1{
    color:#0F172A;
    text-align:center;
    font-weight:700;
}

h3{
    color:#1E3A8A;
}

/* ===============================
Information Cards
=============================== */

.metric-card{
    background:linear-gradient(135deg,#2563EB,#1D4ED8);
    color:white;
    padding:20px;
    border-radius:15px;
    text-align:center;
    box-shadow:0px 5px 12px rgba(0,0,0,0.18);
    margin-bottom:15px;
}

.metric-card h2,
.metric-card h3{
    color:white;
    margin:0;
}

/* ===============================
Predict Button
=============================== */

.stButton > button{
    width:100%;
    height:55px;
    border-radius:12px;
    border:none;
    background:#2563EB;
    color:white;
    font-size:18px;
    font-weight:bold;
}

.stButton > button:hover{
    background:#1E40AF;
    color:white;
}

/* ===============================
Sidebar
=============================== */

[data-testid="stSidebar"]{
    background:#F1F5F9;
}

/* ==========================================
SIDEBAR LABELS
========================================== */

[data-testid="stSidebar"] label{
    color:#0F172A !important;
    font-weight:600;
}

/* Expander titles */

[data-testid="stSidebar"] .streamlit-expanderHeader{
    color:#0F172A !important;
    font-weight:700;
}

/* Selectbox text */

[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"]{
    color:#0F172A !important;
}

/* Number input text */

[data-testid="stSidebar"] input{
    color:#0F172A !important;
}

/* Sidebar markdown */

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span{
    color:#0F172A !important;
}

/* ===============================
Sidebar Input Values
=============================== */

/* Selected value inside selectboxes */
[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    color: #0F172A !important;
}

/* Dropdown placeholder/value */
[data-testid="stSidebar"] div[data-baseweb="select"] span {
    color: #0F172A !important;
}

/* Number input text */
[data-testid="stSidebar"] input[type="number"] {
    color: #0F172A !important;
}

/* Text input (if any) */
[data-testid="stSidebar"] input[type="text"] {
    color: #0F172A !important;
}

/* ===============================
Containers
=============================== */

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

/* ===============================
Success / Error
=============================== */

.stSuccess{
    border-radius:12px;
}

.stError{
    border-radius:12px;
}

.stWarning{
    border-radius:12px;
}

/* ===============================
Footer
=============================== */

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
Determine whether a loan application is likely to be **Approved**
or **Rejected** based on applicant information using a trained
Machine Learning model.
"""
)

st.divider()

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.markdown(
    """
    <h2 style="
        color:#0F172A;
        text-align:center;
        font-weight:bold;
        margin-bottom:20px;">
        📝 Loan Application Form
    </h2>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# PERSONAL INFORMATION
# ==========================================================

with st.sidebar.expander("👤 Personal Information", expanded=True):

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    married = st.selectbox(
        "Marital Status",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["0", "1", "2", "3+"]
    )

    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"]
    )

    self_employed = st.selectbox(
        "Self Employed",
        ["No", "Yes"]
    )

# ==========================================================
# FINANCIAL INFORMATION
# ==========================================================

with st.sidebar.expander("💰 Financial Information", expanded=True):

    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0.0,
        value=5000.0,
        step=100.0
    )

    coapplicant_income = st.number_input(
        "Co-applicant Income",
        min_value=0.0,
        value=0.0,
        step=100.0
    )

    loan_amount = st.number_input(
        "Loan Amount (in thousands)",
        min_value=0.0,
        value=120.0,
        step=1.0
    )

    loan_term = st.selectbox(
        "Loan Amount Term (Days)",
        [12, 36, 60, 84, 120, 180, 240, 300, 360, 480]
    )

    credit_history = st.selectbox(
        "Credit History",
        [1, 0],
        format_func=lambda x: "Good" if x == 1 else "Poor"
    )

# ==========================================================
# PROPERTY INFORMATION
# ==========================================================

with st.sidebar.expander("🏠 Property Information", expanded=True):

    property_area = st.selectbox(
        "Property Area",
        ["Urban", "Semiurban", "Rural"]
    )

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

    "Gender":[gender],
    "Married":[married],
    "Dependents":[dependents],
    "Education":[education],
    "Self_Employed":[self_employed],
    "ApplicantIncome":[applicant_income],
    "CoapplicantIncome":[coapplicant_income],
    "LoanAmount":[loan_amount],
    "Loan_Amount_Term":[loan_term],
    "Credit_History":[credit_history],
    "Property_Area":[property_area]

})

# ==========================================================
# MAKE PREDICTION
# ==========================================================

if predict:

    prediction = model.predict(sample_data)[0]

    try:

        probability = model.predict_proba(sample_data)[0]

        confidence = np.max(probability) * 100

        approval_probability = probability[1] * 100

        rejection_probability = probability[0] * 100

    except:

        confidence = None
        approval_probability = None
        rejection_probability = None

    total_income = applicant_income + coapplicant_income
    
# ==========================================================
# DISPLAY PREDICTION
# ==========================================================

    st.divider()

    col1, col2 = st.columns(2)

    # ======================================================
    # PREDICTION RESULT
    # ======================================================

    with col1:

        if prediction == 1:

            st.success("## ✅ Loan Approved")

            st.write(
                """
                Congratulations! Based on the information provided,
                the applicant is likely to qualify for the requested loan.
                """
            )

        else:

            st.error("## ❌ Loan Rejected")

            st.write(
                """
                Based on the information provided,
                the applicant is unlikely to qualify for the requested loan.
                """
            )

    # ======================================================
    # LOAN APPLICATION SUMMARY
    # ======================================================

    with col2:

        st.markdown(f"""
        <div class="metric-card">
            <h3>📄 Loan Application Summary</h3>
            <h2>₦ {total_income:,.0f}</h2>
            <p>Total Monthly Income</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ======================================================
    # PREDICTION PROBABILITY
    # ======================================================

    if confidence is not None:

        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Prediction Probability</h3>
            <h2>{confidence:.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)

        st.progress(confidence / 100)

    st.divider()

    # ======================================================
    # RECOMMENDATION
    # ======================================================

    st.subheader("💡 Recommendation")

    if prediction == 1:

        st.success(
            """
            The applicant shows characteristics commonly associated
            with successful loan approvals. The application appears
            financially suitable based on the information provided.
            """
        )

    else:

        st.warning(
            """
            Consider improving the applicant's credit history,
            increasing income, reducing the requested loan amount,
            or applying with a stronger financial profile before
            submitting another application.
            """
        )
        
# ==========================================================
# DISCLAIMER
# ==========================================================

st.divider()

st.info(
    """
    **Disclaimer**

    This application is intended for educational and demonstration
    purposes only. Predictions are generated using a Machine Learning
    model trained on historical loan application data and should not
    replace decisions made by financial institutions.
    """
)

# ==========================================================
# FOOTER
# ==========================================================

st.markdown(
"""
<div class="footer">

Developed with ❤️ using <b>Python</b>, <b>Scikit-Learn</b> and <b>Streamlit</b>

</div>
""",
unsafe_allow_html=True
)

            
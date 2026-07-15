import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# LOAD TRAINED PIPELINE
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("Model/model.pkl")

model = load_model()

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main{
    background-color:#f8fafc;
}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

h1{
    color:#0F172A;
    text-align:center;
    font-weight:800;
}

h3{
    color:#1E3A8A;
}

.stButton>button{
    width:100%;
    height:55px;
    border-radius:12px;
    border:none;
    background:#2563EB;
    color:white;
    font-size:18px;
    font-weight:bold;
    transition:0.3s;
}

.stButton>button:hover{
    background:#1D4ED8;
    color:white;
}

div[data-testid="stMetric"]{
    background-color:white;
    border-radius:12px;
    padding:15px;
    box-shadow:0px 3px 8px rgba(0,0,0,0.08);
}

div.stAlert{
    border-radius:12px;
}

hr{
    margin-top:10px;
    margin-bottom:10px;
}

.footer{
    text-align:center;
    color:gray;
    font-size:14px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# APPLICATION HEADER
# ============================================================

st.title("🏦 Loan Approval Prediction System")

st.markdown("""
Welcome to the **Loan Approval Prediction System**.

This application uses a trained **Machine Learning Pipeline**
to estimate whether a loan application is likely to be **Approved**
or **Rejected** based on an applicant's demographic, financial,
and loan-related information.

Fill in the required details from the sidebar and click
**Predict Loan Status** to receive an instant prediction.
""")

st.divider()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📋 Loan Application Form")

st.sidebar.markdown("""
Provide the applicant's information below.
Ensure the details are accurate for the best prediction.
""")

# ============================================================
# APPLICANT INFORMATION
# ============================================================

with st.sidebar.expander("👤 Applicant Information", expanded=True):

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    married = st.selectbox(
        "Married",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Number of Dependents",
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
    
# ============================================================
# FINANCIAL INFORMATION
# ============================================================

with st.sidebar.expander("💰 Financial Information", expanded=True):

    applicant_income = st.number_input(
        "Applicant Monthly Income (₦)",
        min_value=0.0,
        value=5000.0,
        step=500.0,
        format="%.2f"
    )

    coapplicant_income = st.number_input(
        "Co-applicant Monthly Income (₦)",
        min_value=0.0,
        value=1500.0,
        step=500.0,
        format="%.2f"
    )

    credit_history = st.selectbox(
        "Credit History",
        [1.0, 0.0],
        format_func=lambda x: "Good (1)" if x == 1.0 else "Poor (0)"
    )

# ============================================================
# LOAN INFORMATION
# ============================================================

with st.sidebar.expander("🏠 Loan Information", expanded=True):

    loan_amount = st.number_input(
        "Loan Amount (in Thousands)",
        min_value=1.0,
        value=128.0,
        step=1.0,
        format="%.2f"
    )

    loan_term = st.selectbox(
        "Loan Amount Term (Days)",
        [360.0, 180.0, 120.0, 84.0, 60.0, 36.0, 12.0]
    )

    property_area = st.selectbox(
        "Property Area",
        ["Urban", "Semiurban", "Rural"]
    )

st.sidebar.divider()

st.sidebar.info(
    """
    💡 Tips for Better Approval Chances

    • Maintain a good credit history.
    • Higher income generally improves approval chances.
    • Lower loan amounts are usually easier to approve.
    • Stable employment and education may positively influence approval.
    """
)

# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

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

# ============================================================
# MAIN PAGE PREVIEW
# ============================================================

st.subheader("📋 Loan Application Summary")

left, right = st.columns([2, 1])

with left:

    st.dataframe(
        sample_data.T.rename(columns={0: "Applicant Details"}),
        use_container_width=True
    )

with right:

    total_income = applicant_income + coapplicant_income

    st.metric(
        "Total Monthly Income",
        f"₦{total_income:,.2f}"
    )

    st.metric(
        "Requested Loan",
        f"{loan_amount:.0f}K"
    )

    st.metric(
        "Loan Term",
        f"{int(loan_term)} Days"
    )

st.divider()

# ============================================================
# PREDICT BUTTON
# ============================================================

if st.button("🚀 Predict Loan Status"):

    prediction = model.predict(sample_data)[0]

    confidence = None
    approval_probability = None
    rejection_probability = None

    try:

        probabilities = model.predict_proba(sample_data)[0]

        rejection_probability = probabilities[0] * 100
        approval_probability = probabilities[1] * 100

        confidence = np.max(probabilities) * 100

    except Exception:
        pass

    st.divider()

    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    if prediction == 1:

        st.success("### ✅ Loan Approved")

        st.write(
            """
            Congratulations!

            Based on the information provided, the applicant
            has a **high likelihood of receiving loan approval**
            according to the trained Machine Learning model.
            """
        )

        risk_level = "🟢 Low Risk"

    else:

        st.error("### ❌ Loan Rejected")

        st.write(
            """
            Based on the supplied information,
            the applicant is **less likely to receive loan approval**.

            Improving factors such as income,
            credit history and requested loan amount
            may increase future approval chances.
            """
        )

        risk_level = "🔴 High Risk"

    # ========================================================
    # METRICS
    # ========================================================

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Prediction",
            "Approved" if prediction == 1 else "Rejected"
        )

    with col2:

        if confidence is not None:

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

        else:

            st.metric(
                "Confidence",
                "Unavailable"
            )

    with col3:

        st.metric(
            "Risk Level",
            risk_level
        )

    # ========================================================
    # PROBABILITY SECTION
    # ========================================================

    if confidence is not None:

        st.divider()

        st.subheader("📈 Prediction Probability")

        left, right = st.columns(2)

        with left:

            st.write("### ✅ Approval Probability")

            st.progress(float(approval_probability) / 100)

            st.success(
                f"{approval_probability:.2f}%"
            )

        with right:

            st.write("### ❌ Rejection Probability")

            st.progress(float(rejection_probability) / 100)

            st.error(
                f"{rejection_probability:.2f}%"
            )
            
    # ========================================================
    # LOAN APPLICATION INSIGHTS
    # ========================================================

    st.divider()

    st.subheader("📊 Loan Application Insights")

    insight1, insight2 = st.columns(2)

    with insight1:

        st.markdown("### 👤 Applicant Profile")

        st.write(f"**Gender:** {gender}")
        st.write(f"**Marital Status:** {married}")
        st.write(f"**Dependents:** {dependents}")
        st.write(f"**Education:** {education}")
        st.write(f"**Self Employed:** {self_employed}")

    with insight2:

        st.markdown("### 💰 Financial Summary")

        st.write(f"**Applicant Income:** ₦{applicant_income:,.2f}")
        st.write(f"**Co-applicant Income:** ₦{coapplicant_income:,.2f}")
        st.write(f"**Total Income:** ₦{total_income:,.2f}")
        st.write(f"**Loan Amount:** {loan_amount:.0f}K")
        st.write(f"**Loan Term:** {int(loan_term)} Days")
        st.write(f"**Property Area:** {property_area}")

    st.divider()

    # ========================================================
    # SMART RECOMMENDATIONS
    # ========================================================

    st.subheader("💡 Smart Recommendations")

    recommendations = []

    if credit_history == 0:
        recommendations.append(
            "✔ Improve your credit history. Applicants with a good credit history generally have a much higher chance of loan approval."
        )

    if loan_amount > 250:
        recommendations.append(
            "✔ Consider requesting a smaller loan amount to improve approval chances."
        )

    if total_income < 5000:
        recommendations.append(
            "✔ Increasing your monthly income can positively influence loan approval."
        )

    if self_employed == "Yes":
        recommendations.append(
            "✔ Ensure all proof of business income and supporting financial documents are available."
        )

    if dependents == "3+":
        recommendations.append(
            "✔ A high number of dependents may affect affordability assessments."
        )

    if married == "No":
        recommendations.append(
            "✔ Ensure your financial stability is clearly demonstrated during application."
        )

    if education == "Not Graduate":
        recommendations.append(
            "✔ Additional proof of stable income may strengthen your application."
        )

    if len(recommendations) == 0:

        st.success(
            """
            🎉 Excellent!

            Based on the provided information, your application
            demonstrates several positive characteristics commonly
            associated with successful loan approvals.
            """
        )

    else:

        for recommendation in recommendations:
            st.info(recommendation)

    st.divider()

    # ========================================================
    # APPLICATION DETAILS
    # ========================================================

    st.subheader("📝 Submitted Application")

    display_df = sample_data.copy()

    display_df.columns = [
        "Gender",
        "Married",
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

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
    
    # ========================================================
    # DOWNLOAD REPORT
    # ========================================================

    st.divider()

    st.subheader("📥 Download Prediction Report")

    report = pd.DataFrame({

        "Prediction":[
            "Approved" if prediction == 1 else "Rejected"
        ],

        "Confidence (%)":[
            round(confidence,2) if confidence is not None else "N/A"
        ],

        "Applicant Income":[applicant_income],
        "Co-applicant Income":[coapplicant_income],
        "Total Income":[total_income],
        "Loan Amount":[loan_amount],
        "Loan Term":[loan_term],
        "Credit History":[credit_history],
        "Property Area":[property_area]

    })

    csv = report.to_csv(index=False).encode("utf-8")

    st.download_button(

        label="📄 Download Prediction Report",

        data=csv,

        file_name="Loan_Approval_Prediction_Report.csv",

        mime="text/csv"

    )

# ============================================================
# SIDEBAR INFORMATION
# ============================================================

st.sidebar.divider()

st.sidebar.subheader("🤖 Model Information")

st.sidebar.success("""
**Algorithm**

• Logistic Regression

**Pipeline**

• Scikit-Learn Pipeline

**Preprocessing**

• OneHotEncoder
• StandardScaler

**Deployment**

• Streamlit
""")

st.sidebar.divider()

st.sidebar.subheader("📖 About")

st.sidebar.write("""
This application predicts whether a loan application is
likely to be **Approved** or **Rejected** using a trained
Machine Learning model.

The prediction is based on demographic,
financial and loan-related information provided
by the applicant.
""")

st.sidebar.info("""
💡 Note

Predictions generated by this application
are estimates based on historical data and
should not replace decisions made by financial
institutions.
""")

# ============================================================
# FOOTER
# ============================================================

st.divider()

footer_left, footer_center, footer_right = st.columns(3)

with footer_left:

    st.markdown("""
### 🏦 Loan Approval Prediction
Machine Learning Deployment Project
""")

with footer_center:

    st.markdown("""
### 🛠 Technologies

- Python
- Streamlit
- Scikit-Learn
- Pandas
- NumPy
""")

with footer_right:

    st.markdown("""
### 👨‍💻 Developer

Makinde Afolabi Oluwanifemi

Federal University of Agriculture Abeokuta

SIWES Project (2026)
""")

st.markdown(
"""
<div class="footer">

<hr>

<p>
This application was developed for educational and demonstration purposes.
Predictions are generated using a Machine Learning model and should not be considered
as official financial decisions.
</p>

<p>
Built with ❤️ using Streamlit, Scikit-Learn and Python.
</p>

</div>
""",
unsafe_allow_html=True
)                    
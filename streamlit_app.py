import streamlit as st
import yaml

from app.loader import load_models
from app.utils import build_applicant_from_dict
from app.predict import two_stage_predict

with open("config.yaml") as f:
    config = yaml.safe_load(f)

st.set_page_config(page_title="Loan Approval", page_icon="🏦", layout="centered")


@st.cache_resource
def get_models():
    # loaded once and reused across reruns
    return load_models(config)


def fmt_inr(value):
    return f"₹{value:,.0f}"


cls, reg = get_models()
default = config['ui']['default_inputs']

# ---------- header ----------
st.title("🏦 Loan Approval Predictor")
st.caption(
    "Two-stage model: first decides **Approved / Rejected**, "
    "then estimates the **loan amount** for approved applicants."
)

# ---------- sidebar ----------
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        "- **Stage 1:** Random Forest classifier\n"
        "- **Stage 2:** Random Forest regressor\n"
        "- Trained on 4,269 loan applications"
    )
    with st.expander("Features the model expects"):
        try:
            st.write(list(cls.feature_names_in_))
        except Exception:
            st.write("Classifier feature names unavailable")

# ---------- input form ----------
with st.form("applicant_form"):
    st.subheader("👤 Personal details")
    c1, c2, c3 = st.columns(3)
    with c1:
        no_of_dependents = st.number_input(
            "Dependents", min_value=0, max_value=5, step=1,
            value=int(default['no_of_dependents']),
        )
    with c2:
        education = st.selectbox(
            "Education", options=["Graduate", "Not Graduate"],
            index=0 if default['education'] == 'Graduate' else 1,
        )
    with c3:
        self_employed = st.selectbox(
            "Self employed", options=["Yes", "No"],
            index=0 if default['self_employed'] == 'Yes' else 1,
        )

    st.subheader("💰 Loan request")
    c1, c2 = st.columns(2)
    with c1:
        income_annum = st.number_input(
            "Annual income (₹)", min_value=0.0, step=100000.0,
            value=float(default['income_annum']),
        )
        loan_amount = st.number_input(
            "Loan amount requested (₹)", min_value=0.0, step=100000.0,
            value=float(default['loan_amount']),
        )
    with c2:
        loan_term = st.slider(
            "Loan term (years)", min_value=2, max_value=20,
            value=int(default['loan_term']),
        )
        cibil_score = st.slider(
            "CIBIL score", min_value=300, max_value=900,
            value=int(default['cibil_score']),
        )

    st.subheader("🏠 Assets")
    c1, c2 = st.columns(2)
    with c1:
        residential_assets_value = st.number_input(
            "Residential assets (₹)", min_value=0.0, step=100000.0,
            value=float(default['residential_assets_value']),
        )
        commercial_assets_value = st.number_input(
            "Commercial assets (₹)", min_value=0.0, step=100000.0,
            value=float(default['commercial_assets_value']),
        )
    with c2:
        luxury_assets_value = st.number_input(
            "Luxury assets (₹)", min_value=0.0, step=100000.0,
            value=float(default['luxury_assets_value']),
        )
        bank_asset_value = st.number_input(
            "Bank assets (₹)", min_value=0.0, step=100000.0,
            value=float(default['bank_asset_value']),
        )

    submitted = st.form_submit_button("🔍 Predict", type="primary", use_container_width=True)

applicant = {
    'no_of_dependents': no_of_dependents,
    'education': education,
    'self_employed': self_employed,
    'income_annum': income_annum,
    'loan_amount': loan_amount,
    'loan_term': loan_term,
    'cibil_score': cibil_score,
    'residential_assets_value': residential_assets_value,
    'commercial_assets_value': commercial_assets_value,
    'luxury_assets_value': luxury_assets_value,
    'bank_asset_value': bank_asset_value,
}

# ---------- result ----------
if submitted:
    try:
        expected_cols = list(cls.feature_names_in_)
        applicant_df = build_applicant_from_dict(applicant, expected_cols)
        res = two_stage_predict(cls, reg, applicant_df)[0]

        st.divider()
        st.subheader("📊 Result")

        if res['approved'] == 1:
            st.success("✅ **Loan Approved**")
        else:
            st.error("❌ **Loan Rejected**")

        m1, m2 = st.columns(2)
        m1.metric("Approval probability", f"{res['approved_prob']:.1%}")
        if res['reg_pred'] is not None:
            m2.metric(
                "Predicted loan amount",
                fmt_inr(res['reg_pred']),
                delta=fmt_inr(res['reg_pred'] - loan_amount) + " vs requested",
                delta_color="off",
            )
        else:
            m2.metric("Predicted loan amount", "—")

        st.progress(res['approved_prob'])

        if res['approved'] == 0 and cibil_score < 550:
            st.info("💡 A low CIBIL score is the most common reason for rejection in this dataset.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

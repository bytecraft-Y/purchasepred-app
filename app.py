import streamlit as st
import pandas as pd
import joblib

# Load all three models
rf_model = joblib.load('rf_model.pkl')
dt_model = joblib.load('dt_model.pkl')
lr_model = joblib.load('lr_model.pkl')

# Page setup
st.set_page_config(page_title="Purchase Predictor", page_icon="🛍️", layout="centered")

st.title("🛍️ E-commerce Purchase Predictor")
st.markdown("### Choose a model and predict if customer will buy")

st.markdown("---")

# Model Selection
model_choice = st.selectbox(
    "Select Model",
    options=["Random Forest", "Decision Tree", "Logistic Regression"],
    help="Random Forest usually gives best results"
)

# Input fields with better UI
st.subheader("Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("👤 Age", 18, 70, 30)
    session_time = st.slider("⏱️ Session Duration (minutes)", 1, 120, 45)
    pages_viewed = st.slider("📖 Pages Viewed", 1, 50, 12)

with col2:
    items_cart = st.slider("🛒 Items in Cart", 0, 20, 3)
    days_visit = st.slider("📅 Days Since Last Visit", 0, 100, 10)
    discount = st.radio("🎟️ Used Discount?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)

# Predict Button
if st.button("🤖 Predict Purchase", type="primary", use_container_width=True):
    
    # Prepare input
    input_data = pd.DataFrame({
        'Age': [age],
        'Session_Duration_Min': [session_time],
        'Pages_Viewed': [pages_viewed],
        'Items_In_Cart': [items_cart],
        'Days_Since_Last_Visit': [days_visit],
        'Discount_Used': [discount]
    })
    
    # Choose model
    if model_choice == "Random Forest":
        model = rf_model
        model_name = "Random Forest"
    elif model_choice == "Decision Tree":
        model = dt_model
        model_name = "Decision Tree"
    else:
        model = lr_model
        model_name = "Logistic Regression"
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1] * 100 if hasattr(model, "predict_proba") else None

    st.markdown("---")
    
    if prediction == 1:
        st.success(f"### ✅ YES - Likely to PURCHASE!")
        if prob is not None:
            st.metric("Confidence", f"{prob:.1f}%")
    else:
        st.error(f"### ❌ NO - Unlikely to Purchase")
        if prob is not None:
            st.metric("Confidence", f"{100 - prob:.1f}%")

    st.info(f"**Model Used:** {model_name}")

    # Show input summary
    st.subheader("Entered Details")
    st.dataframe(input_data, use_container_width=True)

# Sidebar
with st.sidebar:
    st.header("📌 About This App")
    st.write("This app uses 3 different Machine Learning models to predict customer purchase.")
    st.write("- **Random Forest** (Recommended)")
    st.write("- Decision Tree")
    st.write("- Logistic Regression")
    
    st.markdown("---")
    st.caption("Made for absolute beginners")

st.caption("🛠️ Improved Purchase Prediction App with Multiple Models")

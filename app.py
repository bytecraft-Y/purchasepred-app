import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load('purchase_model.pkl')

# Page Configuration
st.set_page_config(page_title="Purchase Predictor", page_icon="🛍️", layout="centered")

# Header
st.title("🛍️ E-commerce Purchase Predictor")
st.markdown("### Predict whether a customer will buy or not")
st.markdown("---")

# Input Section with better design
st.subheader("Customer Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("👤 Age", min_value=18, max_value=70, value=30, step=1)
    session_time = st.slider("⏱️ Session Duration (minutes)", min_value=1, max_value=120, value=45)
    pages_viewed = st.slider("📖 Pages Viewed", min_value=1, max_value=50, value=12)

with col2:
    items_in_cart = st.slider("🛒 Items in Cart", min_value=0, max_value=20, value=3)
    days_since_visit = st.slider("📅 Days Since Last Visit", min_value=0, max_value=100, value=10)
    discount_used = st.radio("🎟️ Used Discount?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)

# Predict Button - Big and Prominent
if st.button("🔮 Predict if Customer will Buy", type="primary", use_container_width=True):
    
    # Create input data
    input_data = pd.DataFrame({
        'Age': [age],
        'Session_Duration_Min': [session_time],
        'Pages_Viewed': [pages_viewed],
        'Items_In_Cart': [items_in_cart],
        'Days_Since_Last_Visit': [days_since_visit],
        'Discount_Used': [discount_used]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100

    st.markdown("---")
    
    # Show Result with nice styling
    if prediction == 1:
        st.success("### ✅ YES - This customer is **likely to PURCHASE**!")
        st.metric(label="Confidence", value=f"{probability:.1f}%")
    else:
        st.error("### ❌ NO - This customer is **unlikely to purchase**")
        st.metric(label="Confidence", value=f"{100 - probability:.1f}%")

    # Show what user entered
    st.subheader("📋 Customer Details Entered:")
    st.dataframe(input_data, use_container_width=True)

# Sidebar Information
with st.sidebar:
    st.header("About")
    st.write("This app predicts whether a customer will make a purchase based on their behavior.")
    st.write("**Model Used:** Random Forest")
    st.write("**Made for:** Beginners")
    
    st.markdown("---")
    st.caption("Tip: Try different values and see how prediction changes!")

# Footer
st.markdown("---")
#st.caption("🛠️ Simple & Improved Purchase Prediction App | Built for Absolute Beginners")

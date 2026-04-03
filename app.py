import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('purchase_model.pkl')

# Page title and header
st.set_page_config(page_title="Purchase Predictor", page_icon="🛒")
st.title("🛒 E-commerce Purchase Predictor")
st.write("### Enter customer details to predict if they will buy")

st.markdown("---")

# Create two columns for better look
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("👤 Age", min_value=18, max_value=70, value=30, step=1)
    session = st.number_input("⏱️ Session Duration (minutes)", min_value=1, max_value=120, value=45)
    pages = st.number_input("📄 Pages Viewed", min_value=1, max_value=50, value=12)

with col2:
    cart = st.number_input("🛍️ Items in Cart", min_value=0, max_value=20, value=3)
    days = st.number_input("📅 Days Since Last Visit", min_value=0, max_value=100, value=10)
    discount = st.selectbox("🎟️ Used Discount?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Big Predict Button
if st.button("🔮 Predict Purchase", type="primary", use_container_width=True):
    
    # Prepare input data
    input_data = pd.DataFrame({
        'Age': [age],
        'Session_Duration_Min': [session],
        'Pages_Viewed': [pages],
        'Items_In_Cart': [cart],
        'Days_Since_Last_Visit': [days],
        'Discount_Used': [discount]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100   # Probability in %

    # Show result with nice design
    st.markdown("---")
    
    if prediction == 1:
        st.success(f"✅ **YES - Customer is likely to PURCHASE!**")
        st.info(f"Confidence: {probability:.1f}%")
    else:
        st.error(f"❌ **NO - Customer is unlikely to purchase**")
        st.info(f"Confidence: {100 - probability:.1f}%")

    st.write("### Customer Details Entered:")
    st.write(input_data)

# Footer
st.markdown("---")
st.caption("Made with ❤️ for beginners | Simple Purchase Prediction App")

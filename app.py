import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load('purchase_model.pkl')

st.title("🛒 Purchase Prediction App")
st.write("Enter customer details:")

# Simple input boxes
age = st.number_input("Age", 18, 70, 30)
session = st.number_input("Session Duration (minutes)", 1, 120, 45)
pages = st.number_input("Pages Viewed", 1, 50, 12)
cart = st.number_input("Items in Cart", 0, 20, 3)
days = st.number_input("Days Since Last Visit", 0, 100, 10)
discount = st.selectbox("Used Discount?", [0, 1])

# Predict button
if st.button("Predict"):
    data = pd.DataFrame({
        'Age': [age],
        'Session_Duration_Min': [session],
        'Pages_Viewed': [pages],
        'Items_In_Cart': [cart],
        'Days_Since_Last_Visit': [days],
        'Discount_Used': [discount]
    })
    
    pred = model.predict(data)[0]
    
    if pred == 1:
        st.success("✅ YES - Customer will likely BUY!")
    else:
        st.error("❌ NO - Customer will likely NOT buy")
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Silence TensorFlow logs (optional)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Silence Pandas FutureWarning
pd.set_option('future.no_silent_downcasting', True)

# Load the model and preprocessor
try:
    model = tf.keras.models.load_model('spotify_churn_best_ann_model.h5')
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model or preprocessor: {e}")
    st.stop()

# Set up the Streamlit app
st.set_page_config(page_title="Spotitify User Churn Prediction", layout="centered")
st.title("Spotitify User Churn Prediction")
st.markdown("Enter user details to predict if they will churn.")

# Create input widgets for user data
with st.container():
    st.header("User Information")
    age = st.slider("Age", 10, 100, 30)
    gender = st.selectbox("Gender", ['Female', 'Other', 'Male'])
    country = st.selectbox("Country", ['CA', 'DE', 'AU', 'US', 'UK', 'IN', 'FR', 'PK'])
    subscription_type = st.selectbox("Subscription Type", ['Free', 'Family', 'Premium', 'Student'])
    device_type = st.selectbox("Device Type", ['Desktop', 'Web', 'Mobile'])
    listening_time = st.number_input("Listening Time (hours)", min_value=0, value=0)
    songs_played_per_day = st.number_input("Songs Played Per Day", min_value=0, value=0)
    skip_rate = st.slider("Skip Rate (%)", 0, 100, 0)
    ads_listened_per_week = st.number_input("Ads Listened Per Week", min_value=0, value=0)
    offline_listening = st.selectbox("Offline Listening Enabled?", ["No", "Yes"])

    # # Auto-calculate minimum expected total charges
    # expected_min_total = monthly_charges * tenure
    # if total_charges < expected_min_total:
    #     st.warning(f"💡 Total Charges seems low. Expected at least ${expected_min_total:.2f} for {tenure} months.")

    # # Dynamic Senior Citizen based on age (65+)
    # senior_citizen = 1 if age >= 65 else 0
    # partner = st.selectbox("Has Partner?", ["No", "Yes"])
    # dependents = st.selectbox("Has Dependents?", ["No", "Yes"])

    # Create a dictionary from the user inputs — ALL VALUES AS STRINGS OR ORIGINAL TYPE
    input_data = {
        'gender': gender,
        'age': age,
        'country': country,
        'subscription type': subscription_type,
        'device type': device_type,
        'listening time': listening_time,
        'songs played per day': songs_played_per_day,
        'skip rate': skip_rate,
        'ads listened per week': ads_listened_per_week,
        'offline listening': offline_listening,
    }

# Convert input data to a DataFrame
input_df = pd.DataFrame([input_data])

# Button to make a prediction
if st.button("Predict Churn"):
    try:
        # ✅ CRITICAL: DO NOT MANUALLY ENCODE — Let preprocessor handle it
        # The preprocessor expects raw categories (strings) for categorical columns
        # and raw numbers for numerical columns — just like during training.

        # Ensure numerical columns are float (optional safety)
        numerical_cols = ['listening_time', 'songs_played_per_day', 'skip_rate', 'ads_listened_per_week']
        for col in numerical_cols:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)

        # ✅ Apply the preprocessor — it will OneHotEncode categorical and Scale numerical
        input_processed = preprocessor.transform(input_df)

        # Make prediction
        prediction = model.predict(input_processed)
        churn_probability = prediction[0][0]

        # Display result
        st.subheader("Prediction Result")
        if churn_probability > 0.5:
            st.error(f"🚨 Prediction: This customer is likely to churn. (Probability: {churn_probability:.2%})")
        else:
            st.success(f"✅ Prediction: This customer is not likely to churn. (Probability: {churn_probability:.2%})")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.exception(e)  # For debugging — remove in production

st.markdown("---")
st.caption("💡 Tip: Adjust inputs to see how churn probability changes.")
st.info("This app is a demonstration of a machine learning model deployed with Streamlit.")

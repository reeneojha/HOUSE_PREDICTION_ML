import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
model = XGBRegressor()
model.load_model("house_price_model.json")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Price Prediction")
st.write(
    "Predict house prices using an XGBoost Machine Learning model."
)
st.sidebar.header("Enter Property Details")
city = st.sidebar.selectbox(
    "City",
    ["Bangalore", "Hyderabad", "Mumbai", "Nagpur", "Pune"]
)
locality = st.sidebar.selectbox(
    "Locality Tier",
    ["Budget", "Mid", "Premium"]
)
bhk = st.sidebar.number_input(
    "BHK",
    min_value=1,
    max_value=10,
    value=2
)

bathrooms = st.sidebar.number_input(
    "Bathrooms",
    min_value=1,
    max_value=10,
    value=2
)


furnishing = st.sidebar.selectbox(
    "Furnishing",
    [
        "Fully-Furnished",
        "Semi-Furnished",
        "Unfurnished"
    ]
)


parking = st.sidebar.selectbox(
    "Parking Available",
    ["Yes", "No"]
)


lift = st.sidebar.selectbox(
    "Lift Available",
    ["Yes", "No"]
)


gated = st.sidebar.selectbox(
    "Gated Society",
    ["Yes", "No"]
)
if st.button("Predict Price"):

    input_data = {
        "BHK": bhk,
        "Bathrooms": bathrooms,
        "Super_Area_sqft": 950.50,
        "Carpet_Area_sqft": 715.55,
        "Floor_No": 7,
        "Total_Floors": 15,
        "Property_Age_years": 8,
        "Parking": 1 if parking == "Yes" else 0,
        "Lift": 1 if lift == "Yes" else 0,
        "Gated_Society": 1 if gated == "Yes" else 0,
        "Distance_to_Metro_km": 1.83,
        "Distance_to_CityCenter_km": 14.01,
        "Nearby_School_km": 2.80,
        "Nearby_Hospital_km": 3.00,
        "Crime_Rate_Index": 41.80
    }

    input_df = pd.DataFrame([input_data])
    input_df["City"] = city
    input_df["Locality_Tier"] = locality
    input_df["Furnishing"] = furnishing
    input_df = pd.get_dummies(
        input_df,
        drop_first=True
    )
    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )
    prediction = model.predict(input_df)
    price = prediction[0]
    st.success(
        f"Estimated House Price: Rs {price:,.0f}"
    )

    st.info(
        "Model Used: XGBoost Regressor"
    )
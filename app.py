import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)
model = XGBRegressor()
model.load_model("house_price_model.json")
feature_columns = joblib.load("feature_columns.pkl")
def format_indian(number):
    number = int(round(number))
    is_negative = number < 0
    number = abs(number)

    s = str(number)
    if len(s) <= 3:
        formatted = s
    else:
        last3 = s[-3:]
        rest = s[:-3]
        parts = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        formatted = ",".join(parts) + "," + last3

    return ("-" if is_negative else "") + formatted


def to_words_indian(number):
    """Return a short human label like '45.3 Lakh' or '1.2 Crore'."""
    number = abs(number)
    if number >= 1_00_00_000:  # 1 crore
        return f"{number / 1_00_00_000:.2f} Crore"
    elif number >= 1_00_000:  # 1 lakh
        return f"{number / 1_00_000:.2f} Lakh"
    else:
        return f"{number:,.0f}"
st.title("🏠 House Price Prediction")
st.caption("Estimate residential property prices instantly using a trained XGBoost regression model.")
st.divider()

st.sidebar.header(" Property Details")
st.sidebar.write("Fill in the details below, then hit **Predict Price**.")

st.sidebar.subheader(" Location")
city = st.sidebar.selectbox("City", ["Bangalore", "Hyderabad", "Mumbai", "Nagpur", "Pune"])
locality = st.sidebar.selectbox("Locality Tier", ["Budget", "Mid", "Premium"])

st.sidebar.subheader(" Layout")
bhk = st.sidebar.number_input("BHK", min_value=1, max_value=10, value=2)
bathrooms = st.sidebar.number_input("Bathrooms", min_value=1, max_value=10, value=2)
furnishing = st.sidebar.selectbox(
    "Furnishing",
    ["Fully-Furnished", "Semi-Furnished", "Unfurnished"]
)

st.sidebar.subheader(" Amenities")
parking = st.sidebar.selectbox("Parking Available", ["Yes", "No"])
lift = st.sidebar.selectbox("Lift Available", ["Yes", "No"])
gated = st.sidebar.selectbox("Gated Society", ["Yes", "No"])

st.sidebar.divider()
predict_clicked = st.sidebar.button("🔮 Predict Price", use_container_width=True, type="primary")
left, right = st.columns(2, gap="large")

with left:
    with st.container(border=True):
        st.subheader("Selected Configuration")

        rows = [
            ("City", city, "Locality Tier", locality),
            ("BHK", bhk, "Bathrooms", bathrooms),
            ("Furnishing", furnishing, "Parking", parking),
            ("Lift", lift, "Gated Society", gated),
        ]

        for label1, val1, label2, val2 in rows:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**{label1}**")
                st.write(val1)
            with c2:
                st.markdown(f"**{label2}**")
                st.write(val2)
            st.write("")  # small spacer

with right:
    with st.container(border=True):
        st.subheader("Prediction")

        if predict_clicked:
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
            input_df = pd.get_dummies(input_df, drop_first=True)
            input_df = input_df.reindex(columns=feature_columns, fill_value=0)

            prediction = model.predict(input_df)
            price = prediction[0]

            st.metric("Estimated House Price", f"₹ {format_indian(price)}")
        else:
            st.info(" Set your property details in the sidebar and click **Predict Price** to see the estimate.")

st.divider()
st.caption("Built with Streamlit & XGBoost")
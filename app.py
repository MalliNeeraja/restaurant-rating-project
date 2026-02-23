#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib

st.set_page_config(page_title="Restaurant Rating Predictor", layout="wide")

st.title("🍽️ Restaurant Rating Analysis & Prediction")
st.write("Cognifyz Data Science Internship Project")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Dataset.csv")
    return df

df = load_data()

# Load Saved Model
@st.cache_resource
def load_model():
    model = joblib.load("restaurant_rating_model.pkl")
    return model

model = load_model()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Basic cleaning
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# Feature engineering
df["Restaurant_Name_Length"] = df["Restaurant Name"].apply(len)
df["Address_Length"] = df["Address"].apply(len)
df["Has_Table_Booking_Flag"] = df["Has Table booking"].map({"Yes": 1, "No": 0})
df["Has_Online_Delivery_Flag"] = df["Has Online delivery"].map({"Yes": 1, "No": 0})

# Encode categorical columns
encoder = LabelEncoder()
for col in ["City", "Cuisines", "Currency"]:
    df[col] = encoder.fit_transform(df[col])

features = [
    "City",
    "Cuisines",
    "Average Cost for two",
    "Price range",
    "Votes",
    "Has_Table_Booking_Flag",
    "Has_Online_Delivery_Flag",
    "Restaurant_Name_Length",
    "Address_Length"
]

X = df[features]
y = df["Aggregate rating"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


st.subheader("Model Performance")
preds = model.predict(X_test)
st.write("R2 Score:", r2_score(y_test, preds))

# Sidebar for prediction
st.sidebar.header("Predict Restaurant Rating")

city = st.sidebar.number_input("City (encoded value)", min_value=0)
cuisine = st.sidebar.number_input("Cuisines (encoded value)", min_value=0)
cost = st.sidebar.number_input("Average Cost for Two", min_value=0)
price_range = st.sidebar.number_input("Price Range", min_value=0)
votes = st.sidebar.number_input("Votes", min_value=0)
table_booking = st.sidebar.selectbox("Has Table Booking?", [0, 1])
online_delivery = st.sidebar.selectbox("Has Online Delivery?", [0, 1])
name_len = st.sidebar.number_input("Restaurant Name Length", min_value=1)
addr_len = st.sidebar.number_input("Address Length", min_value=1)

if st.sidebar.button("Predict Rating"):
    input_data = np.array([[city, cuisine, cost, price_range, votes,
                             table_booking, online_delivery, name_len, addr_len]])
    prediction = model.predict(input_data)[0]
    st.success(f"⭐ Predicted Restaurant Rating: {prediction:.2f}")


# In[ ]:





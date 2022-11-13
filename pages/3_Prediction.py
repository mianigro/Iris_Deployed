import streamlit as st
import numpy as np
from make_pred import make_prediction
import json
import pandas as pd
import plotly.express as px

# Setup data from csv
df = pd.read_csv("iris.data",
                 header=None,
                 names=["Sepal Length", "Sepal Width",
                        "Petal Length", "Petal Width", "Species"])


# Setup title page
st.set_page_config(page_title="Prediction")
st.header("Prediction - Iris Dataset")
st.markdown("Using XGBoost, make predictions for what classification the species data should fall into. "
            "The prediction will appear on the graphs below to intuit how the prediction was made.")
st.sidebar.header("Make Prediction")

sep_len = st.sidebar.text_input("Sepal Length")
sep_wid = st.sidebar.text_input("Sepal Width")
pet_len = st.sidebar.text_input("Petal Length")
pet_wid = st.sidebar.text_input("Petal Width")
make_pred = st.sidebar.button("Predict")

# Load json for encoding
with open('encoder.json') as json_file:
    data = json.load(json_file)


# Managing input data
p1 = ["", "", "", ""]

plot1 = px.scatter(
    df,
    x="Petal Length",
    y="Petal Width",
    title="Petal Length vs Petal Width",
    color="Species")

plot2 = px.scatter(
    df,
    x="Sepal Length",
    y="Petal Length",
    title="Sepal Length vs Petal Length",
    color="Species")

# Making a prediction and displaying data
if make_pred:
    p1 = [float(sep_len), float(sep_wid), float(pet_len), float(pet_wid)]
    x = np.array([p1])
    row = {"Sepal Length": [float(sep_len)],
           "Sepal Width": [float(sep_wid)],
           "Petal Length": [float(pet_len)],
           "Petal Width": [float(pet_wid)]}

    p1_df = pd.DataFrame(row)
    species_pred = make_prediction(x)

    st.subheader(f"Predicted Species: {species_pred}")
    plot1.add_scatter(x=p1_df["Petal Length"], y=p1_df["Petal Width"])
    plot2.add_scatter(x=p1_df["Sepal Length"], y=p1_df["Petal Length"])

st.plotly_chart(plot1)
st.plotly_chart(plot2)

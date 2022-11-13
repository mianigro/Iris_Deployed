import streamlit as st
import pandas as pd
import plotly.express as px

# Setup data
df = pd.read_csv("iris.data",
                 header=None,
                 names=["Sepal Length", "Sepal Width",
                        "Petal Length", "Petal Width", "Species"])


# Make page
st.set_page_config(page_title="Iris Dataset")
st.header("Comparison - Iris Dataset")
st.markdown("Explore the variables to understand the relationship between them and how they relate to the species. "
            "As patterns emerge, we can intuit how the XGBoost makes decisions in classifying data.")
st.sidebar.header("Variable Comparison")

# Setting graph to display
options = st.sidebar.radio("Select comparison",
                           options=["Sepal Length Vs Sepal Width",
                                    "Petal Length Vs Petal Width",
                                    "Sepal Length Vs Petal Width",
                                    "Sepal Width Vs Petal Length"])

if options == "Sepal Length Vs Sepal Width":
    plot = px.scatter(
        df,
        x="Sepal Length",
        y="Sepal Width",
        color="Species",
        title=options)

elif options == "Petal Length Vs Petal Width":
    plot = px.scatter(
        df,
        x="Petal Length",
        y="Petal Width",
        color="Species",
        title=options)

elif options == "Sepal Length Vs Petal Width":
    plot = px.scatter(
        df,
        x="Sepal Length",
        y="Petal Width",
        color="Species",
        title=options)

elif options == "Sepal Width Vs Petal Length":
    plot = px.scatter(
        df,
        x="Sepal Width",
        y="Petal Length",
        color="Species",
        title=options)

st.plotly_chart(plot)

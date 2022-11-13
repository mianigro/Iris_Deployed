import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

# Setup data
df = pd.read_csv("iris.data",
                 header=None,
                 names=["Sepal Length", "Sepal Width",
                        "Petal Length", "Petal Width", "Species"])


# Make page
st.set_page_config(page_title="Iris Dataset")
st.header("Values - Iris Dataset")
st.markdown("Explore how each individual variable is related to each species. "
            "We can intuit patterns with the individual values and understand how the data is used to perform classifcation.")
st.sidebar.header("Individual Values")

# Setting graph to display
options = st.sidebar.radio("Select values",
                           options=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"])

show_df = df.filter(items=[options, "Species"])

plot1 = px.histogram(
    show_df,
    x=show_df[options],
    title=f"{options} Histogram",
    nbins=30,
    color="Species")

st.plotly_chart(plot1)



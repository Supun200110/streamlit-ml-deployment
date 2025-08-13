# ===============================
# app.py
# ===============================

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("quality/notebooks/model_training.ipynb")

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Title and description
st.title("🍷 Wine Quality Prediction (Regression)")
st.write("""
This app predicts wine quality based on its chemical properties.
""")

# Sidebar navigation
menu = ["Dataset Overview", "Visualisations", "Predict", "Model Performance"]
choice = st.sidebar.selectbox("Navigation", menu)

if choice == "Dataset Overview":
    st.subheader("Dataset Overview")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    st.write(df.head())
    st.write("Summary Statistics")
    st.write(df.describe())

elif choice == "Visualisations":
    st.subheader("Visualisations")
    # Correlation heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot(plt)
    plt.clf()
    
    # Histogram of target
    sns.histplot(df['quality'], kde=True, bins=10)
    st.pyplot(plt)
    plt.clf()

elif choice == "Predict":
    st.subheader("Make a Prediction")
    # Get user input
    top_features = ['alcohol', 'volatile acidity', 'sulphates', 'citric acid', 'density', 'total sulfur dioxide']
    input_data = []
    for col in top_features:
     val = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
     input_data.append(val)
    # Prediction
    if st.button("Predict Quality"):
        input_array = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_array)[0]
        st.success(f"Predicted Wine Quality: {prediction:.2f}")

elif choice == "Model Performance":
      st.subheader("Model Performance")
import json
with open("quality/metrics.json", "r") as f:
        metrics = json.load(f)
     
        st.write(f"**Best Model:** {metrics['BestModel']}")
        st.write("### Performance Metrics")
        st.table(pd.DataFrame(metrics).drop(columns=['BestModel'], errors='ignore'))

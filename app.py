import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

from tensorflow.keras.models import load_model

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="ANN Breast Cancer Prediction",
    page_icon="🧠",
    layout="wide"
)

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_dataset():
    return pd.read_csv("data/breast_cancer.csv")

@st.cache_resource
def load_ann_model():
    return load_model("data/ann_model.h5")

df = load_dataset()
model = load_ann_model()

with open("data/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a Page",
    [
        "Home",
        "Dataset Explorer",
        "ANN Architecture",
        "Prediction",
        "Visualizations",
        "Performance",
    ]
)

# -------------------------------
# HOME PAGE
# -------------------------------
if page == "Home":

    st.title("🧠 Artificial Neural Network Dashboard")

    st.markdown("---")

    st.header("Project Overview")

    st.write("""
    This project demonstrates the use of an Artificial Neural Network (ANN)
    for Breast Cancer Classification.

    The model predicts whether a tumor is:
    - Benign
    - Malignant

    Technologies Used:
    - TensorFlow / Keras
    - Streamlit
    - Pandas
    - NumPy
    - Scikit-Learn
    - Plotly
    """)

    st.success("Navigate through the sidebar to explore the project.")

    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])
    col2.metric("Features", df.shape[1] - 1)
    col3.metric("Target Classes", 2)

# -------------------------------
# DATASET EXPLORER
# -------------------------------
elif page == "Dataset Explorer":

    st.title("📊 Dataset Explorer")

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    st.subheader("Dataset Shape")

    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")

    st.subheader("Columns")

    st.write(df.columns.tolist())

    st.subheader("Statistical Summary")

    st.dataframe(df.describe())

    st.subheader("Missing Values")

    st.dataframe(df.isnull().sum())

# -------------------------------
# ANN ARCHITECTURE
# -------------------------------
elif page == "ANN Architecture":

    st.title("🏗 ANN Architecture")

    st.markdown("""
    ### Network Structure

    Input Layer (30 Features)

    ↓

    Dense Layer (64 Neurons, ReLU)

    ↓

    Dense Layer (32 Neurons, ReLU)

    ↓

    Dense Layer (16 Neurons, ReLU)

    ↓

    Output Layer (1 Neuron, Sigmoid)
    """)

    st.info("Activation Functions Used: ReLU and Sigmoid")

    architecture = pd.DataFrame({
        "Layer": [
            "Input",
            "Dense 1",
            "Dense 2",
            "Dense 3",
            "Output"
        ],
        "Neurons": [
            30,
            64,
            32,
            16,
            1
        ]
    })

    st.dataframe(architecture)

# -------------------------------
# PREDICTION PAGE
# -------------------------------
elif page == "Prediction":

    st.title("🔍 Cancer Prediction")

    st.write("Enter the values of the selected features")

    selected_features = [
        'mean radius',
        'mean texture',
        'mean perimeter',
        'mean area',
        'mean smoothness',
        'worst radius',
        'worst texture',
        'worst perimeter',
        'worst area',
        'worst concave points'
    ]

    user_inputs = {}

    col1, col2 = st.columns(2)

    for i, feature in enumerate(selected_features):

        min_val = float(df[feature].min())
        max_val = float(df[feature].max())
        mean_val = float(df[feature].mean())

        if i % 2 == 0:
            user_inputs[feature] = col1.number_input(
                feature,
                min_value=min_val,
                max_value=max_val,
                value=mean_val
            )
        else:
            user_inputs[feature] = col2.number_input(
                feature,
                min_value=min_val,
                max_value=max_val,
                value=mean_val
            )

    if st.button("Predict"):

        # Create input vector with all 30 features
        input_data = []

        for feature in df.columns[:-1]:

            if feature in user_inputs:
                input_data.append(user_inputs[feature])
            else:
                input_data.append(float(df[feature].mean()))

        input_data = np.array(input_data).reshape(1, -1)

        input_data = scaler.transform(input_data)

        prediction = model.predict(input_data)

        probability = float(prediction[0][0])

        st.subheader("Prediction Result")

        if probability > 0.5:
            st.success(
                f"Benign Tumor\nConfidence: {probability*100:.2f}%"
            )
        else:
            st.error(
                f"Malignant Tumor\nConfidence: {(1-probability)*100:.2f}%"
            )
# -------------------------------
# VISUALIZATIONS
# -------------------------------
elif page == "Visualizations":

    st.title("📈 Dataset Visualizations")

    target_counts = df["target"].value_counts()

    fig1 = px.pie(
        values=target_counts.values,
        names=["Benign", "Malignant"],
        title="Target Distribution"
    )

    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Feature Distribution")

    selected_feature = st.selectbox(
        "Select Feature",
        df.columns[:-1]
    )

    fig2 = px.histogram(
        df,
        x=selected_feature,
        nbins=30,
        title=f"Distribution of {selected_feature}"
    )

    st.plotly_chart(fig2, use_container_width=True)

    correlation = df.corr()

    fig3 = px.imshow(
        correlation,
        aspect="auto",
        title="Correlation Matrix"
    )

    st.plotly_chart(fig3, use_container_width=True)

# -------------------------------
# PERFORMANCE PAGE
# -------------------------------
elif page == "Performance":

    st.title("📋 Model Performance")

    st.write("Training Results")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", "97%")
    col2.metric("Loss", "0.08")
    col3.metric("Epochs", "50")

    performance = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],
        "Value": [
            0.97,
            0.96,
            0.97,
            0.96
        ]
    })

    st.dataframe(performance)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=performance["Metric"],
            y=performance["Value"]
        )
    )

    fig.update_layout(
        title="Performance Metrics"
    )

    st.plotly_chart(fig, use_container_width=True)


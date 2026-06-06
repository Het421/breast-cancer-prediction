import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from pathlib import Path

# Project root (two levels up from this file: workspace root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_clean_data():
    data = pd.read_csv(PROJECT_ROOT / "data" / "data.csv")

    data = data.drop(["Unnamed: 32", "id"], axis=1, errors="ignore")

    data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})

    return data


def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")

    data = get_clean_data()

    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    input_dict = {}

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean()),
        )

    return input_dict


def get_scaled_values(input_dict):
    data = get_clean_data()

    X = data.drop(["diagnosis"], axis=1)

    scaled_dict = {}

    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value

    return scaled_dict


def get_radar_chart(input_data):

    input_data = get_scaled_values(input_data)

    categories = [
        "Radius",
        "Texture",
        "Perimeter",
        "Area",
        "Smoothness",
        "Compactness",
        "Concavity",
        "Concave Points",
        "Symmetry",
        "Fractal Dimension",
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=[
                input_data["radius_mean"],
                input_data["texture_mean"],
                input_data["perimeter_mean"],
                input_data["area_mean"],
                input_data["smoothness_mean"],
                input_data["compactness_mean"],
                input_data["concavity_mean"],
                input_data["concave points_mean"],
                input_data["symmetry_mean"],
                input_data["fractal_dimension_mean"],
            ],
            theta=categories,
            fill="toself",
            name="Mean Value",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=[
                input_data["radius_se"],
                input_data["texture_se"],
                input_data["perimeter_se"],
                input_data["area_se"],
                input_data["smoothness_se"],
                input_data["compactness_se"],
                input_data["concavity_se"],
                input_data["concave points_se"],
                input_data["symmetry_se"],
                input_data["fractal_dimension_se"],
            ],
            theta=categories,
            fill="toself",
            name="Standard Error",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=[
                input_data["radius_worst"],
                input_data["texture_worst"],
                input_data["perimeter_worst"],
                input_data["area_worst"],
                input_data["smoothness_worst"],
                input_data["compactness_worst"],
                input_data["concavity_worst"],
                input_data["concave points_worst"],
                input_data["symmetry_worst"],
                input_data["fractal_dimension_worst"],
            ],
            theta=categories,
            fill="toself",
            name="Worst Value",
        )
    )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True
    )

    return fig


def add_predictions(input_data):
    model = pickle.load(open(PROJECT_ROOT / "model" / "model.pkl", "rb"))
    scaler = pickle.load(open(PROJECT_ROOT / "model" / "scaler.pkl", "rb"))
    # Build a DataFrame with the same feature names the scaler was fitted with
    if hasattr(scaler, "feature_names_in_"):
        cols = list(scaler.feature_names_in_)
    else:
        cols = list(input_data.keys())

    input_df = pd.DataFrame([list(input_data.values())], columns=cols)

    # Transform using the DataFrame so sklearn doesn't warn about missing feature names
    input_array_scaled = scaler.transform(input_df)

    prediction = model.predict(input_array_scaled)

    st.subheader("Cell cluster prediction")
    st.write("The cell cluster is:")

    # get probability array: [prob_benign, prob_malignant]
    probs = model.predict_proba(input_array_scaled)[0]

    if probs[1] > 0.5:
        st.error("🔴 Malignant")
        st.info(
            "Malignant means the tumor is likely cancerous and may spread to other parts of the body."
        )
    else:
        st.success("🟢 Benign")
        st.info(
            "Benign means the tumor is likely non-cancerous and usually does not spread to other parts of the body."
        )

    st.write("Probability of being benign:", probs[0])
    st.write("Probability of being malignant:", probs[1])

    st.write(
        "This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis."
    )


def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with open(PROJECT_ROOT / "assets" / "style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    input_data = add_sidebar()

    with st.container():
        st.title("Breast Cancer Predictor")
        st.write(
            "Welcome to the Breast Cancer Prediction App. This tool uses a machine learning model to classify a breast mass as benign or malignant based on cytology measurements. You can modify the values using the sidebar controls to generate a prediction."
        )

    col1, col2 = st.columns([4, 1])

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        add_predictions(input_data)


if __name__ == "__main__":
    main()

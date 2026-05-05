import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="Iris Classifier", page_icon="🌸", layout="centered")

MODEL_PATH = "model.pkl"

@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    else:
        iris = load_iris()
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(iris.data, iris.target)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
    return model

model = load_or_train_model()

st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: #f8fafc; }
    h1 { color: #e2e8f0; font-family: 'Inter', sans-serif; text-align: center; font-weight: 700; margin-bottom: 2rem; }
    .stButton>button { background: linear-gradient(135deg, #8b5cf6, #ec4899); color: white; border-radius: 8px; padding: 0.5rem 2rem; border: none; width: 100%; margin-top: 1rem; }
    .metric-card { background: rgba(30, 41, 59, 0.7); padding: 2rem; border-radius: 16px; text-align: center; margin-top: 2rem; }
    .species-title { background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.5rem; font-weight: 800; }
</style>
""", unsafe_allow_html=True)

st.title("🌸 Iris AI Classifier")

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.8)
    sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.0)
with col2:
    petal_length = st.slider("Petal Length", 1.0, 7.0, 3.8)
    petal_width = st.slider("Petal Width", 0.1, 2.5, 1.2)

if st.button("Analyze & Predict 🌼"):
    prediction = model.predict(np.array([[sepal_length, sepal_width, petal_length, petal_width]]))
    species = load_iris().target_names[prediction[0]]
    st.markdown(f'<div class="metric-card"><h4>Prediction Result</h4><div class="species-title">Iris {species.capitalize()}</div></div>', unsafe_allow_html=True)

import streamlit as st
import joblib
import numpy as np

st.title("Prediksi Obesitas Berdasarkan Gaya Hidup")

# Load model & fitur
model = joblib.load("model_obesity.pkl")
fitur = joblib.load("fitur.pkl")

# Buat input sesuai fitur
input_data = []

for nama in fitur:
    if "Age" in nama or "Height" in nama or "Weight" in nama:
        nilai = st.number_input(f"{nama}", step=0.1)
    elif "FCVC" in nama or "NCP" in nama or "CH2O" in nama:
        nilai = st.slider(f"{nama}", 0.0, 3.0, step=0.1)
    else:
        nilai = st.selectbox(f"{nama}", [0, 1])
    input_data.append(nilai)

# Prediksi
if st.button("Prediksi"):
    hasil = model.predict([input_data])
    st.success(f"Hasil Prediksi: {hasil[0]}")

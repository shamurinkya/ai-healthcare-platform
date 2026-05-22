import streamlit as st
import numpy as np
import pandas as dp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model("models/ecg_model.h5")

st.title("ECG Arrhythmia Detection AI")

uploaded_file = st.file_uploader("Upload ECG CSV File", type=["csv"])

if uploaded_file is not None:
    
    data =pd.read_csv(uploded_file, header=None)

    st.write(data.head())

    sample = data.values[0].reshape(1, 187, 7)

    prediction = model.predict(sample)

    predicted_class = np.argmax(prediction)

    classes = {
        0: "Normal Beat",
        1: "Supraventricular Beat",
        2: "Ventricular Beat",
        3: "Fusion Beat",
        4: "Unknown Beat"
    }

    st.success(f"Prediction: {classes[predicted_class]}")
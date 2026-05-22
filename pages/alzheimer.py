import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

model = load_model("models/alzheimer_model.h5")

classes = [
    "Mild Demented",
    "Moderate Demented",
    "Non Demented",
    "Very Mild Demented"
]

st.title("Alzheimer MRI Detection AI")

st.write("Upload an MRI image to predict Alzheimer stage.")

uploaded_file = st.file_uploader(
    "Choose MRI Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    img = Image.open(uploaded_file)

    st.image(img, caption="Uploaded MRI Image", use_container_width=True)

    img = img.resize((150,150))

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = img_array / 255.0

    prediction = model.predict(img_array)

    predicted_class = classes[np.argmax(prediction)]

    confidence = np.max(prediction) * 100

    st.success(f"Prediction: {predicted_class}")

    st.info(f"Confidence: {confidence:.2f}%")

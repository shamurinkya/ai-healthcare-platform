import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model("models/brain_tumor_model.h5")

st.title("Brain Tumor Detection App")

uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(150,150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)

    classes = ["glioma", "meningioma", "no_tumor", "pituitary"]
    result = classes[np.argmax(prediction)]

    st.image(uploaded_file, caption="Upload Image", use_column_width=True)
    st.write("Prediction:", result)
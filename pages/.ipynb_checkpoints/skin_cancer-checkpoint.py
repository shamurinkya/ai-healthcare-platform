import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model("models/skin_cancer_model.h5")
classes = ["Benign", "Malignant"]

st.set_page_config(
    page_title="Skin Cancer Detection AI",
    page_icon="🩺",
    layout="centered"
)

st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }

    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #1f4e79;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: gray;
        margin-bottom: 30px;
    }

    .result {
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">🩺 Skin Cancer Detection AI</p>', unsafe_allow_html=True)

st.markdown(
    '<p class="subtitle">Upload a skin image to detect cancer type</p>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload Skin Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    img = Image.open(uploaded_file)

    st.image(
        img,
        caption="Uploaded Image",
        use_container_width=True
    )

    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)

    result = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success("Prediction Completed ✅")

    st.markdown(
        f"""
        <div class="result">
            Prediction: {result}<br><br>
            Confidence: {confidence:.2f}%
        </div>
        """,
        unsafe_allow_html=True
    )

st.sidebar.title("About")

st.sidebar.info(
    """
    This AI model detects skin cancer from skin lesion images.

    Classes:
    - Benign
    - Malignant

    Built using TensorFlow & Streamlit.
    """
)
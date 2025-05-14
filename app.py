import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64

# ---------- App Config ----------
st.set_page_config(page_title="Mango Leaf Classifier", layout="centered")

# ---------- Custom CSS ----------
def local_css():
    st.markdown("""
        <style>
            .main {
                background-color: #f4f4f4;
                padding: 2rem;
                border-radius: 12px;
                box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
            }
            .header {
                font-size: 2.5rem;
                color: #1a5d1a;
                text-align: center;
                margin-bottom: 1rem;
                font-weight: bold;
            }
            .footer {
                text-align: center;
                font-size: 0.9rem;
                margin-top: 2rem;
                color: gray;
            }
            .result {
                background-color: #dff0d8;
                color: #3c763d;
                padding: 1rem;
                border-radius: 8px;
                font-size: 1.2rem;
                text-align: center;
                font-weight: bold;
                margin-top: 1rem;
            }
            .error {
                background-color: #f2dede;
                color: #a94442;
                padding: 1rem;
                border-radius: 8px;
                font-size: 1.2rem;
                text-align: center;
                font-weight: bold;
                margin-top: 1rem;
            }
        </style>
    """, unsafe_allow_html=True)

local_css()

# ---------- Load Model ----------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("efficientnetb3_model.keras")

model = load_model()

# ---------- Class Names ----------
CLASS_NAMES = ['Banganapalli', 'Kesar', 'Mallika', 'Mangilal', 'Mankurad',
               'Monserrate', 'Neelam', 'Sindhu', 'Totapuri']

# ---------- Descriptions ----------
CLASS_DESCRIPTIONS = {
    'Banganapalli': "Large, golden-yellow mango with smooth skin and fiberless, sweet pulp. Popular in Andhra Pradesh; early-season variety ideal for fresh consumption.",
    'Kesar': "Medium-sized mango with bright saffron pulp. Sweet, aromatic, and widely used in desserts. Grown in Gujarat and favored for export.",
    'Mallika': "Hybrid of Neelam and Dasheri. Long-shaped, bright orange mango with fiberless, sweet pulp. Known for disease resistance and excellent taste.",
    'Mangilal': "Local Goan variety with greenish-yellow skin and mildly sweet-tangy flavor. Grown mainly for regional markets and home use.",
    'Mankurad': "Iconic Goan variety. Small to medium-sized, golden-yellow mango with juicy, sweet, fiberless pulp. Highly prized and harvested early.",
    'Monserrate': "Traditional Goan mango with a sweet-sour flavor and distinct shape. Medium pulp fiber and primarily consumed fresh within Goa.",
    'Neelam': "Late-season variety with reddish-yellow skin and tangy-sweet aroma. Slightly fibrous and suitable for long-distance transport.",
    'Sindhu': "Seedless/small-seed mango developed in Maharashtra. Sweet, fiberless, and ideal for pulp extraction. Early-mid season harvest.",
    'Totapuri': "Parrot-beak-shaped mango with firm, mildly tangy pulp. Thick skin and minimal fiber. Widely used in juice and pulp industry."
}

# ---------- Preprocess Function ----------
def preprocess_image(image):
    image = image.resize((1536, 1536))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# ---------- UI Layout ----------
st.markdown('<div class="header">🌿 Mango Leaf Classifier</div>', unsafe_allow_html=True)
st.markdown("Upload a mango leaf image to detect the variety.")

uploaded_file = st.file_uploader("Choose a mango leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        st.markdown(
    f"""
    <div style="display: flex; justify-content: center; align-items: center;">
        <img src="data:image/png;base64,{base64.b64encode(uploaded_file.getvalue()).decode()}"
             width="302" height="302" style="border-radius: 10px;" />
    </div>
    """,
    unsafe_allow_html=True
)



        with st.spinner('Classifying...'):
            processed = preprocess_image(image)
            prediction = model.predict(processed)
            predicted_class = CLASS_NAMES[np.argmax(prediction)]
            confidence = 100 * np.max(prediction)

        st.markdown(f'<div class="result">Predicted Class: {predicted_class}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result">Confidence: {confidence:.2f}%</div>', unsafe_allow_html=True)

        description = CLASS_DESCRIPTIONS.get(predicted_class, "Description not available.")
        st.markdown(f'<div class="result">Description: {description}</div>', unsafe_allow_html=True)

    except Exception as e:
        st.markdown(f'<div class="error">Error: {str(e)}</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">Developed as a Final Year Project | 2025</div>', unsafe_allow_html=True)

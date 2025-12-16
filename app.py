import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# ================= CONFIG =================
MODEL_PATH = "1HHegBF8GNpA-FcnDYdD4HZBvSBs1-P5J"   # <-- your .h5 file name
IMG_SIZE = 224

CLASS_NAMES = [
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented"
]

# ================= PAGE SETUP =================
st.set_page_config(
    page_title="Alzheimer's Disease Prediction",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Alzheimer's Disease Prediction")
st.markdown("AI-based MRI image classification")
st.divider()

# ================= LOAD MODEL =================
@st.cache_resource
def load_alzheimer_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()
    return load_model(MODEL_PATH)

model = load_alzheimer_model()

# ================= IMAGE UPLOAD =================
uploaded_file = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)

# ================= PREDICTION =================
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded MRI Image", use_container_width=True)

    with st.spinner("Analyzing MRI image..."):
        # Preprocessing
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Prediction
        prediction = model.predict(img_array)[0]
        predicted_index = np.argmax(prediction)
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = prediction[predicted_index]

    st.divider()
    st.subheader("🩺 Prediction Result")

    # Dementia or not
    if predicted_class == "NonDemented":
        st.success(f"🟢 **{predicted_class}**")
    else:
        st.error(f"🔴 **{predicted_class}**")

    st.write(f"**Confidence:** {confidence * 100:.2f}%")

    # ================= PROBABILITIES =================
    st.subheader("📊 Class Probabilities")
    for cls, prob in zip(CLASS_NAMES, prediction):
        st.write(f"{cls}")
        st.progress(float(prob))

# ================= FOOTER =================
st.divider()
st.caption("⚠️ This AI tool is for educational purposes only and not a medical diagnosis.")

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import gdown

# --- Configuration based on your Alzheimer's Notebook ---
# 1. Model File: Replace this with the actual path or Google Drive ID/setup.
# If you run Streamlit locally and the model is in the same directory:
# MODEL_PATH = "alzheimer.h5"
# Or if you use the gdown method (replace with your file's ID):
MODEL_ID = "1HHegBF8GNpA-FcnDYdD4HZBvSBs1-P5J"  # 🔴🔴 REPLACE THIS ID 🔴🔴
MODEL_PATH = "alzheimer_model.h5"

# 2. Image Size: Used for resizing input images.
IMG_SIZE = 224

# 3. Class Names: From your notebook output (Cell 8)
CLASS_NAMES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Alzheimer's MRI Diagnosis",
    page_icon="🧠",
    layout="centered"
)

# ================== CUSTOM CSS (Hospital Style) ==================
st.markdown("""
<style>
body { background-color: #f5f7fa; }
.main { background-color: #ffffff; border-radius: 15px; padding: 20px; }
h1 { color: #0b5394; font-weight: 700; }
.result-box { padding: 20px; border-radius: 12px; margin-top: 15px; }
/* Custom colors for multi-class results */
.demented { background-color: #FB4549; border-left: 6px solid #d9534f; color: white; }
.non-demented { background-color: #009900; border-left: 6px solid #2ecc71; color: white; }
.footer { text-align: center; color: gray; font-size: 13px; }
</style>
""", unsafe_allow_html=True)

# ================== LOAD MODEL ==================
@st.cache_resource
def load_alzheimer_model():
    """Loads the Keras model, downloading from Drive if needed."""
    if not os.path.exists(MODEL_PATH) and MODEL_ID != "YOUR_ALZHEIMER_MODEL_DRIVE_ID":
        try:
            st.warning(f"Downloading model from Google Drive (ID: {MODEL_ID})... This may take a moment.")
            url = f"https://drive.google.com/uc?id={MODEL_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
            st.success("Model downloaded successfully.")
        except Exception as e:
            st.error(f"Failed to download model from Google Drive: {e}")
            st.stop()
    elif not os.path.exists(MODEL_PATH):
        st.error(f"Model file '{MODEL_PATH}' not found. Please place it in the application directory or set the correct MODEL_ID.")
        st.stop()

    # Load the model
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading Keras model: {e}")
        st.stop()

# --- Load the model ---
model = load_alzheimer_model()

# ================== HEADER ==================
st.markdown("<h1 style='text-align:center;'>🧠 Alzheimer's Disease Classification</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-assisted MRI image analysis (4 classes)</p>", unsafe_allow_html=True)
st.markdown("---")

# ================== IMAGE INPUT ==================
uploaded_image = st.file_uploader(
    "Upload Brain MRI Image (JPG/PNG)",
    type=["jpg", "jpeg", "png"]
)

# ================== PREDICTION ==================
if uploaded_image is not None:
    img = Image.open(uploaded_image).convert("RGB")
    st.image(img, caption="Uploaded MRI Image", use_container_width=True)

    with st.spinner('Analyzing image...'):
        # Preprocess the image
        img_resized = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict (returns probabilities for 4 classes)
        # Output shape will be (1, 4)
        predictions = model.predict(img_array, verbose=0)[0]
        
        # Get the index of the highest probability
        predicted_index = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = predictions[predicted_index]

    st.markdown("## 🔬 Diagnostic Result")

    # Determine status for result box styling
    is_demented = predicted_class in ['MildDemented', 'ModerateDemented', 'VeryMildDemented']
    result_style = "demented" if is_demented else "non-demented"
    status_text = "Dementia Detected" if is_demented else "Non-Demented"
    
    # ================== RESULT BOX ==================
    st.markdown(
        f"""
        <div class="result-box {result_style}">
        <h3>⚠️ {status_text}</h3>
        <p>Predicted Stage: <b>{predicted_class}</b></p>
        <p>Confidence: <b>{confidence*100:.2f}%</b></p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # ================== PROBABILITY BREAKDOWN ==================
    st.markdown("### 📊 Probability Breakdown")
    
    # Create a dictionary of results for sorting
    results = dict(zip(CLASS_NAMES, predictions))
    
    # Display results for all classes
    for class_name, probability in sorted(results.items(), key=lambda item: item[1], reverse=True):
        st.markdown(f"**{class_name}**")
        col_bar, col_metric = st.columns([4, 1])
        with col_bar:
            # Highlight the progress bar for the highest probability
            if class_name == predicted_class:
                st.progress(probability, text=f"**{probability*100:.2f}%**")
            else:
                st.progress(probability, text=f"{probability*100:.2f}%")
        with col_metric:
             st.metric("", f"{probability*100:.2f}%")

# ================== FOOTER ==================
st.markdown("---")
st.markdown(
    "<div class='footer'>⚕️ AI-assisted system | Not a substitute for professional diagnosis</div>",
    unsafe_allow_html=True
)

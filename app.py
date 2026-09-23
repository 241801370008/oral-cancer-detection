import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Oral Cancer Detection",
    page_icon="🦷",
    layout="centered"
)

# Load trained model
model = tf.keras.models.load_model(
    "models/oral_cancer_final.keras"
)

# Title
st.title("🦷 Oral Cancer Detection")
st.write("Upload an oral image to get an AI-based classification result.")

# Upload image
uploaded_file = st.file_uploader(
    "Choose an oral image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.image(
        image,
        caption="Uploaded Image",
        width="stretch"
    )

    # Resize image
    image_resized = image.resize((224, 224))

    # Convert image to NumPy array
    image_array = np.array(image_resized)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction
    prediction = model.predict(
        image_array,
        verbose=0
    )[0][0]

    # Determine class
    if prediction >= 0.5:
        predicted_class = "Oral Cancer"
        confidence = prediction
    else:
        predicted_class = "Normal"
        confidence = 1 - prediction

    # Display result
    st.subheader("Prediction Result")

    st.write(
        "**Predicted Class:**",
        predicted_class
    )

    st.write(
        "**Confidence:**",
        f"{confidence * 100:.2f}%"
    )

    # Disclaimer
    st.warning(
        "⚠️ This is an educational AI image-classification "
        "result, not a medical diagnosis."
    )
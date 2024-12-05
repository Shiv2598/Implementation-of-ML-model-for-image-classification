import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("mobilenetv2_model.h5")

# Class labels (adjust according to your dataset)
CLASS_NAMES = ["Class1", "Class2"]  # Replace with actual class names

# Streamlit app
st.title("Image Classification with MobileNetV2")
st.write("Upload an image and let the model classify it.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file).resize((224, 224))
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Convert image to array and normalize
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    # Predict
    predictions = model.predict(image_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    st.write(f"**Prediction**: {predicted_class}")
    st.write(f"**Confidence**: {confidence:.2f}%")

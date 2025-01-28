import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import zipfile

# Directory where the model is stored (Adjust this based on your repo structure)
model_dir = "./models"  # This should be where your `.h5` file is stored
model_h5_name = "plant_disease_model.h5"

# Ensure the model is available in the repo
if not os.path.exists(os.path.join(model_dir, model_h5_name)):
    st.write("Model not found. Please upload or fetch the model.")
else:
    # Load the model
    model = load_model(os.path.join(model_dir, model_h5_name))
    img_size = 224

    # Define the class names for prediction
    class_names = {0: 'Apple___Apple_scab', 1: 'Apple___Black_rot', ...}  # Your class names

    # Prediction function
    def predict_image(image):
        img = image.resize((img_size, img_size))  # Resize image to match model input
        img_array = np.array(img) / 255.0        # Normalize image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        prediction = model.predict(img_array)
        return np.argmax(prediction), np.max(prediction)

    # Streamlit app interface
    st.title("Plant Disease Classification App")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Processing...")

        class_idx, confidence = predict_image(image)
        class_name = class_names.get(class_idx, "Unknown")

        st.write(f"Predicted Class: {class_name}")
        st.write(f"Confidence: {confidence:.2f}")

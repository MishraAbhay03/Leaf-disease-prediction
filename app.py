import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import zipfile

# Directory where the model is stored (Adjust this based on your repo structure)
model_dir = "./"  # This should be where your `.h5` file is stored
model_h5_name = "plant_disease_model.h5"

# Ensure the model is available in the repo
if not os.path.exists(os.path.join(model_dir, model_h5_name)):
    st.write("Model not found. Please upload or fetch the model.")
else:
    # Load the model
    model = load_model(os.path.join(model_dir, model_h5_name))
    img_size = 224

    # Define the class names for prediction
    class_names = {0: 'Apple___Apple_scab',
 1: 'Apple___Black_rot',
 2: 'Apple___Cedar_apple_rust',
 3: 'Apple___healthy',
 4: 'Blueberry___healthy',
 5: 'Cherry_(including_sour)___Powdery_mildew',
 6: 'Cherry_(including_sour)___healthy',
 7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 8: 'Corn_(maize)___Common_rust_',
 9: 'Corn_(maize)___Northern_Leaf_Blight',
 10: 'Corn_(maize)___healthy',
 11: 'Grape___Black_rot',
 12: 'Grape___Esca_(Black_Measles)',
 13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 14: 'Grape___healthy',
 15: 'Orange___Haunglongbing_(Citrus_greening)',
 16: 'Peach___Bacterial_spot',
 17: 'Peach___healthy',
 18: 'Pepper,_bell___Bacterial_spot',
 19: 'Pepper,_bell___healthy',
 20: 'Potato___Early_blight',
 21: 'Potato___Late_blight',
 22: 'Potato___healthy',
 23: 'Raspberry___healthy',
 24: 'Soybean___healthy',
 25: 'Squash___Powdery_mildew',
 26: 'Strawberry___Leaf_scorch',
 27: 'Strawberry___healthy',
 28: 'Tomato___Bacterial_spot',
 29: 'Tomato___Early_blight',
 30: 'Tomato___Late_blight',
 31: 'Tomato___Leaf_Mold',
 32: 'Tomato___Septoria_leaf_spot',
 33: 'Tomato___Spider_mites Two-spotted_spider_mite',
 34: 'Tomato___Target_Spot',
 35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 36: 'Tomato___Tomato_mosaic_virus',
 37: 'Tomato___healthy'}  # Your class names

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

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

# Load your trained model
model = tf.keras.models.load_model('path/to/your/model.h5')

# Assuming 'data_classes' is a list or dictionary mapping indices to class names
data_classes = ['Class 1', 'Class 2', 'Class 3']  # Update with actual class names

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize to match the model input shape
    img_array = np.array(img) / 255.0  # Normalize the image
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

# Streamlit app
st.title("Leaf Disease Prediction")
st.write("Upload a leaf image to predict its disease.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    # Load and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(processed_image)
    predicted_class = data_classes[np.argmax(prediction[0])]
    confidence = np.max(prediction[0]) * 100  # Confidence percentage

    # Display the prediction
    st.write(f"Predicted Class: {predicted_class} (Confidence: {confidence:.2f}%)")

import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image

# Load your trained model
model = load_model('model.h5')  # Replace 'model.h5' with the path to your model

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((150, 150))  # Resize to the input size of the model
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image
    return img_array

# Streamlit app
st.title("Image Classification with VGG16")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    
    # Preprocess the image
    img_array = preprocess_image(img)
    
    # Make a prediction
    prediction = model.predict(img_array)
    
    # Display the prediction
    if prediction[0] > 0.5:
        st.write("This is a Dog image")
    else:
        st.write("This is a Cat image")

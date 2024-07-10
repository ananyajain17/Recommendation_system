import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Load CSS file
def load_css(file_name):
    with open(file_name, 'r') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load custom CSS
load_css('style.css')
# Load your data and model here
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Function to save uploaded file
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except Exception as e:
        st.error(f"Error: {e}")
        return 0

# Function for feature extraction
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Function to recommend similar images
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# Streamlit app title and file upload
st.title('Fashion Recommender System')
uploaded_file = st.file_uploader("Choose an image")

# Display results based on uploaded file
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # Display uploaded image
        display_image = Image.open(uploaded_file)
        st.image(display_image, caption='Uploaded Image', use_column_width=True)

        # Extract features and recommend similar images
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
        indices = recommend(features, feature_list)

        # Display recommended images in columns
        st.subheader('Recommended Images:')
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.image(filenames[indices[0][0]], use_column_width=True)
        with col2:
            st.image(filenames[indices[0][1]], use_column_width=True)
        with col3:
            st.image(filenames[indices[0][2]], use_column_width=True)
        with col4:
            st.image(filenames[indices[0][3]], use_column_width=True)
        with col5:
            st.image(filenames[indices[0][4]], use_column_width=True)
    else:
        st.error("Error occurred while uploading file.")

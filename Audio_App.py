import streamlit as st
import librosa
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from PIL import Image
import os

MODEL_PATH = r'C:\Users\Mihir\Audio_Classification\final_audio_model.keras'
LABELENCODER_PATH = r'C:\Users\Mihir\Audio_Classification\final_audio_classes.pkl'
IMAGE_FOLDER = r'C:\Users\Mihir\Audio_Classification\images'

model = load_model(MODEL_PATH)
with open(LABELENCODER_PATH, 'rb') as file:
    labelencoder = pickle.load(file)

def predict_audio_label(audio_data, sample_rate):
    mfccs_features = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features, axis=1)

    mfccs_scaled_features = mfccs_scaled_features.reshape(1, 40, 1)

    y_pred = model.predict(mfccs_scaled_features)

    y_pred_classes = np.argmax(y_pred, axis=1)

    prediction_class = labelencoder.inverse_transform(y_pred_classes)

    return prediction_class[0]

st.title("Audio Classification")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3","mpeg","ogg"])

if uploaded_file is not None:
    
    audio_data, sample_rate = librosa.load(uploaded_file, res_type='kaiser_fast') 

    predicted_label = predict_audio_label(audio_data, sample_rate)

    st.markdown(f"<h3>Predicted Label: {predicted_label}</h3>", unsafe_allow_html=True)

    st.audio(uploaded_file, format='audio/wav') 
    
    image_path = os.path.join(IMAGE_FOLDER, f"{predicted_label}.jpg")
    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, width=400)
    else:
        st.write("No image found for the predicted label.")

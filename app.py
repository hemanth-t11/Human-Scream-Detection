import streamlit as st
import tempfile
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from audio_features import extract_features
from pydub import AudioSegment
from io import BytesIO
from PIL import Image
import os

st.title(" Human Scream Detection")

# Load trained model
model = joblib.load("scream_classifier.pkl")

# Upload audio file
audio_file = st.file_uploader("Upload a .wav or .mp3 file", type=["wav", "mp3"])

# Display model performance metrics
if os.path.exists("model_metrics.csv"):
    st.subheader(" Model Performance")
    metrics_df = pd.read_csv("model_metrics.csv")
    st.dataframe(metrics_df)

# Show confusion matrix image
if os.path.exists("confusion_matrix.png"):
    st.subheader(" Confusion Matrix")
    st.image(Image.open("confusion_matrix.png"), use_container_width=True)

# Predict scream/non-scream from uploaded audio
if audio_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        if audio_file.name.endswith(".mp3"):
            sound = AudioSegment.from_file(BytesIO(audio_file.read()), format="mp3")
            sound.export(tmp.name, format="wav")
        else:
            tmp.write(audio_file.read())
            tmp.flush()

        try:
            features = extract_features(tmp.name)
            prediction = model.predict([features])[0]
            st.success(f" Detected Sound: **{prediction.upper()}**")

        except Exception as e:
            st.error(f" Error: {e}")

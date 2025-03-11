import streamlit as st
import pickle
import os
import json
import numpy as np
import io
from scipy.io.wavfile import write

# Paths
DATA_FOLDER = "results/"  # Folder containing .pkl files
SAVE_FILE = "similarity_ratings.json"
SAMPLE_RATE = 22050  # Adjust based on your data

# Load or initialize ratings
if os.path.exists(SAVE_FILE):
    with open(SAVE_FILE, "r") as f:
        ratings = json.load(f)
else:
    ratings = {}

# Get all pickle files
pkl_files = sorted([f for f in os.listdir(DATA_FOLDER) if f.endswith(".pkl")])

st.title("Sound Similarity Survey")

def array_to_wav(audio_array, sample_rate=22050):
    """Convert a NumPy array to WAV binary data."""
    audio_array = np.array(audio_array, dtype=np.float32)  # Ensure correct type
    audio_array = (audio_array * 32767).astype(np.int16)  # Scale to 16-bit PCM
    wav_buffer = io.BytesIO()
    write(wav_buffer, sample_rate, audio_array)
    return wav_buffer.getvalue()

for pkl_file in pkl_files:
    # Load data from pickle file
    with open(os.path.join(DATA_FOLDER, pkl_file), "rb") as f:
        data = pickle.load(f)
    
    # Convert JAX types to normal Python floats
    target_sound = [[float(element) for element in sublist] for sublist in data["target_sound"]]
    output_sound = [[float(element) for element in sublist] for sublist in data["output_sound"]]

    # Flatten if necessary (assuming mono audio)
    target_sound = np.array(target_sound).flatten()
    output_sound = np.array(output_sound).flatten()

    # Convert list to WAV binary
    target_wav = array_to_wav(target_sound, SAMPLE_RATE)
    output_wav = array_to_wav(output_sound, SAMPLE_RATE)

    st.subheader(f"Survey for {pkl_file}")
    
    # Play target sound
    st.write("🔊 Target Sound:")
    st.audio(target_wav, format="audio/wav")
    
    # Play output sound
    st.write("🔊 Output Sound:")
    st.audio(output_wav, format="audio/wav")

    # Load previous rating if available
    prev_rating = ratings.get(pkl_file, 50)

    # Unique slider key (Fixes Duplicate ID Error)
    similarity = st.slider(
        f"Similarity Score for {pkl_file}",  
        0, 100, prev_rating, key=pkl_file
    )

    # Save the rating
    ratings[pkl_file] = similarity

# Save ratings when button is clicked
if st.button("Save Ratings"):
    with open(SAVE_FILE, "w") as f:
        json.dump(ratings, f)
    st.success("Ratings saved!")


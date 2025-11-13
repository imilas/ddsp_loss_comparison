import streamlit as st
import pickle
import os
import json
import numpy as np
import io
import random
from scipy.io.wavfile import write
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str,nargs="?",default="hearing_test/in_domain", help="Path to folder containing .pkl files")
args = parser.parse_args()


DATA_FOLDER = args.data_folder

# Paths
SAVE_FILE = DATA_FOLDER+"/"+"similarity_ratings.json"
print(SAVE_FILE)
SAMPLE_RATE = 44100  # Adjust based on your data

# Load or initialize ratings in session state
if "ratings" not in st.session_state:
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE, "r") as f:
            st.session_state.ratings = json.load(f)
    else:
        st.session_state.ratings = {}

ratings = st.session_state.ratings  # Alias for easier use

# Get all pickle files
all_pkl_files = sorted([f for f in os.listdir(DATA_FOLDER) if f.endswith(".pkl")])

# Extract program numbers from filenames (assuming format: *_<program_num>_*.pkl)
program_numbers = sorted(set(f.split("_")[-2] for f in all_pkl_files))

# Dropdown to select program number
selected_program = st.selectbox("Select Program Number:", program_numbers)

# Filter files by selected program number
filtered_pkl_files = [f for f in all_pkl_files if f"_{selected_program}_" in f]

# Cache the shuffled order in session state
shuffle_key = f"shuffled_order_{selected_program}"
if shuffle_key not in st.session_state:
    st.session_state[shuffle_key] = random.sample(filtered_pkl_files, len(filtered_pkl_files))  # Shuffle once

pkl_files = st.session_state[shuffle_key]  # Use cached shuffled list

st.title("Sound Similarity Survey")

@st.cache_data
def load_pkl_file(filepath):
    """Load a .pkl file and convert data to NumPy arrays."""
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    
    # Convert JAX types to normal Python floats
    target_sound = np.array([[float(element) for element in sublist] for sublist in data["target_sound"]]).flatten()
    output_sound = np.array([[float(element) for element in sublist] for sublist in data["output_sound"]]).flatten()

    return target_sound, output_sound

def array_to_wav(audio_array, sample_rate=44100):
    """Convert a NumPy array to WAV binary data."""
    audio_array = np.array(audio_array, dtype=np.float32)  # Ensure correct type
    audio_array = (audio_array * 32767).astype(np.int16)  # Scale to 16-bit PCM
    wav_buffer = io.BytesIO()
    write(wav_buffer, sample_rate, audio_array)
    return wav_buffer.getvalue()

def save_ratings():
    """Save ratings to JSON file."""
    with open(SAVE_FILE, "w") as f:
        json.dump(st.session_state.ratings, f)

# Survey for each file
for pkl_file in pkl_files:
    # Load data using cache
    target_sound, output_sound = load_pkl_file(os.path.join(DATA_FOLDER, pkl_file))

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
    prev_rating = ratings.get(pkl_file, 3)

    # Unique slider key
    slider_key = f"slider_{pkl_file}"

    # Create a slider and automatically update the ratings on change
    similarity = st.slider(
        f"Similarity Score for ",
        1, 5, prev_rating, key=slider_key, step=1
    )

    # Save only if the value changes
    if similarity != ratings.get(pkl_file):
        ratings[pkl_file] = similarity
        save_ratings()  # Auto-save on change

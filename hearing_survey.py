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
SAMPLE_RATE = 44100  # Adjust based on your data

# Get all pickle files (limit to 5 for testing)
all_pkl_files = sorted([f for f in os.listdir(DATA_FOLDER) if f.endswith(".pkl")])
# all_pkl_files = all_pkl_files[0:5]

# Load or initialize ratings
if "ratings" not in st.session_state:
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE, "r") as f:
            st.session_state.ratings = json.load(f)
    else:
        st.session_state.ratings = {}

# Ensure all files exist in session state with default value 2
for pkl_file in all_pkl_files:
    if pkl_file not in st.session_state.ratings:
        st.session_state.ratings[pkl_file] = 2

# Extract program numbers from filenames (assuming format: *_<program_num>_*.pkl)
program_numbers = sorted(set(int(f.split("_")[-2]) for f in all_pkl_files))

# Dropdown to select program number
selected_program = st.selectbox("Select Program Number:", program_numbers)

# Filter files by selected program number
pkl_files = [f for f in all_pkl_files if f"_{selected_program}_" in f]

st.title("Sound Similarity Survey")

@st.cache_data
def load_audio_data(pkl_file):
    """Load and process audio data from a .pkl file (cached)."""
    with open(os.path.join(DATA_FOLDER, pkl_file), "rb") as f:
        data = pickle.load(f)

    # Convert JAX types to normal Python floats
    target_sound = [[float(element) for element in sublist] for sublist in data["target_sound"]]
    output_sound = [[float(element) for element in sublist] for sublist in data["output_sound"]]

    # Flatten if necessary (assuming mono audio)
    target_sound = np.array(target_sound).flatten()
    output_sound = np.array(output_sound).flatten()

    return target_sound, output_sound

def array_to_wav(audio_array, sample_rate=44100):
    """Convert a NumPy array to WAV binary data."""
    audio_array = np.array(audio_array, dtype=np.float32)  # Ensure correct type
    audio_array = (audio_array * 32767).astype(np.int16)    # Scale to 16-bit PCM
    wav_buffer = io.BytesIO()
    write(wav_buffer, sample_rate, audio_array)
    return wav_buffer.getvalue()

def save_ratings():
    """Save the current ratings to the JSON file."""
    with open(SAVE_FILE, "w") as f:
        json.dump(st.session_state.ratings, f)

# Iterate over files and display each survey element
for pkl_file in pkl_files:
    target_sound, output_sound = load_audio_data(pkl_file)  # Cached loading

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

    # Get previous rating
    prev_rating = st.session_state.ratings[pkl_file]

    # Unique slider key to avoid conflicts
    similarity = st.slider(
        f"Similarity Score for {pkl_file}",  
        0, 4, prev_rating, key=f"slider_{pkl_file}", step=1
    )
    
    # Update session state and save immediately when slider changes
    st.session_state.ratings[pkl_file] = similarity
    save_ratings()  # Auto-save on slider update


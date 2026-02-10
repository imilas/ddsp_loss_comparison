import streamlit as st
import pickle, os, json, numpy as np, io, random
from scipy.io.wavfile import write
import argparse, sys

# --- args (works with: streamlit run app.py -- --data_folder path) ---
parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str, nargs="?", default="hearing_test/in_domain")
args, _ = parser.parse_known_args(sys.argv[1:])
DATA_FOLDER = args.data_folder
SAMPLE_RATE = 44100

st.set_page_config(page_title="Sound Similarity Survey", layout="centered")
st.title("Sound Similarity Survey")

SAVE_FILE = os.path.join(DATA_FOLDER, "similarity_ratings.json")

# ----------------- Caching helpers -----------------

@st.cache_data
def list_pkl_files(data_folder: str):
    files = sorted([f for f in os.listdir(data_folder) if f.endswith(".pkl")])
    programs = sorted(set(f.split("_")[-2] for f in files))
    return files, programs

@st.cache_data
def load_sounds(path: str, mtime: float):
    # mtime is only used as a cache key so cache invalidates if file changes
    with open(path, "rb") as f:
        d = pickle.load(f)
    # vectorized, no Python loops:
    tgt = np.asarray(d["target_sound"], dtype=np.float32).ravel()
    out = np.asarray(d["output_sound"], dtype=np.float32).ravel()
    return tgt, out

@st.cache_data
def to_wav_bytes(x: np.ndarray, sr: int, key: str):
    # key should be a small hash (e.g., path+mtime); we pass it from caller
    x16 = np.clip(x, -1, 1)          # safety
    x16 = (x16 * 32767).astype(np.int16)
    buf = io.BytesIO()
    write(buf, sr, x16)
    return buf.getvalue()

# ----------------- Init ratings & file list -----------------

if "ratings" not in st.session_state:
    st.session_state.ratings = json.load(open(SAVE_FILE)) if os.path.exists(SAVE_FILE) else {}

ratings = st.session_state.ratings
all_files, program_numbers = list_pkl_files(DATA_FOLDER)

selected_program = st.selectbox("Program:", program_numbers)
filtered = [f for f in all_files if f"_{selected_program}_" in f]

# One-time shuffle per program
shuffle_key = f"shuffle_{selected_program}"
if shuffle_key not in st.session_state:
    st.session_state[shuffle_key] = random.sample(filtered, len(filtered))

pkl_files = st.session_state[shuffle_key]

# ----------------- Pagination: one item at a time -----------------

if "idx" not in st.session_state:
    st.session_state.idx = 0

col_prev, col_pos, col_next = st.columns([1,2,1])
with col_prev:
    if st.button("◀ Prev", use_container_width=True) and st.session_state.idx > 0:
        st.session_state.idx -= 1
with col_next:
    if st.button("Next ▶", use_container_width=True) and st.session_state.idx < len(pkl_files)-1:
        st.session_state.idx += 1
with col_pos:
    st.write(f"{st.session_state.idx+1} / {len(pkl_files)}")

current = pkl_files[st.session_state.idx]
path = os.path.join(DATA_FOLDER, current)
mtime = os.path.getmtime(path)

# Load & convert ONLY the current file
tgt, out = load_sounds(path, mtime)
tgt_wav = to_wav_bytes(tgt, SAMPLE_RATE, key=f"{current}-tgt-{mtime}")
out_wav = to_wav_bytes(out, SAMPLE_RATE, key=f"{current}-out-{mtime}")

# st.subheader(current)
st.write("🔊 Target")
st.audio(tgt_wav, format="audio/wav")
st.write("🔊 Output")
st.audio(out_wav, format="audio/wav")

# Use a form so the slider doesn't cause constant reruns while dragging
with st.form(key=f"form-{current}"):
    prev = ratings.get(current, 3)
    score = st.slider("Similarity (1–5)", 1, 5, prev, step=1)
    submitted = st.form_submit_button("Save rating")
    if submitted:
        ratings[current] = score
        with open(SAVE_FILE, "w") as f:
            json.dump(ratings, f, indent=2)
        st.success("Saved.")



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
    with open(path, "rb") as f:
        d = pickle.load(f)
    tgt = np.asarray(d["target_sound"], dtype=np.float32).ravel()
    out = np.asarray(d["output_sound"], dtype=np.float32).ravel()
    return tgt, out

@st.cache_data
def to_wav_bytes(x: np.ndarray, sr: int, key: str):
    x16 = np.clip(x, -1, 1)
    x16 = (x16 * 32767).astype(np.int16)
    buf = io.BytesIO()
    write(buf, sr, x16)
    return buf.getvalue()

# ----------------- Init ratings & file list -----------------

if "ratings" not in st.session_state:
    st.session_state.ratings = json.load(open(SAVE_FILE)) if os.path.exists(SAVE_FILE) else {}

ratings = st.session_state.ratings
all_files, program_numbers = list_pkl_files(DATA_FOLDER)

# ----------------- State init -----------------

if "idx" not in st.session_state:
    st.session_state.idx = 0

if "selected_program" not in st.session_state:
    st.session_state.selected_program = program_numbers[0] if program_numbers else ""

if "prev_program" not in st.session_state:
    st.session_state.prev_program = st.session_state.selected_program

def persist_current_rating_for_program(program: str):
    """Save current slider for the current file of the given program (if possible)."""
    if not program:
        return

    shuffle_key = f"shuffle_{program}"
    if shuffle_key not in st.session_state:
        # program hasn't been initialized/shuffled yet
        return

    pkl_files = st.session_state[shuffle_key]
    if not pkl_files:
        return

    idx = st.session_state.get("idx", 0)
    idx = max(0, min(idx, len(pkl_files) - 1))
    current_file = pkl_files[idx]

    slider_key = f"score_{current_file}"
    if slider_key in st.session_state:
        ratings[current_file] = int(st.session_state[slider_key])
        with open(SAVE_FILE, "w") as f:
            json.dump(ratings, f, indent=2)

def on_program_change():
    # Save rating from the program we are leaving
    old_program = st.session_state.prev_program
    persist_current_rating_for_program(old_program)

    # Reset position for new program
    st.session_state.idx = 0

    # Remember this as the "previous" program for next change
    st.session_state.prev_program = st.session_state.selected_program

# Program dropdown with autosave-on-change
st.selectbox(
    "Program:",
    program_numbers,
    key="selected_program",
    on_change=on_program_change,
)

selected_program = st.session_state.selected_program
filtered = [f for f in all_files if f"_{selected_program}_" in f]

# One-time shuffle per program
shuffle_key = f"shuffle_{selected_program}"
if shuffle_key not in st.session_state:
    st.session_state[shuffle_key] = random.sample(filtered, len(filtered))

pkl_files = st.session_state[shuffle_key]

# ----------------- Prev/Next autosave -----------------

def persist_current_rating():
    persist_current_rating_for_program(st.session_state.selected_program)

def go_prev():
    persist_current_rating()
    if st.session_state.idx > 0:
        st.session_state.idx -= 1

def go_next():
    persist_current_rating()
    if st.session_state.idx < len(pkl_files) - 1:
        st.session_state.idx += 1

col_prev, col_pos, col_next = st.columns([1, 2, 1])
with col_prev:
    st.button("◀ Prev", use_container_width=True, on_click=go_prev, disabled=(st.session_state.idx <= 0))
with col_next:
    st.button("Next ▶", use_container_width=True, on_click=go_next, disabled=(st.session_state.idx >= len(pkl_files) - 1))
with col_pos:
    st.write(f"{st.session_state.idx+1} / {len(pkl_files)}")

if not pkl_files:
    st.warning("No .pkl files found for this program.")
    st.stop()

current = pkl_files[st.session_state.idx]
path = os.path.join(DATA_FOLDER, current)
mtime = os.path.getmtime(path)

# Load & convert ONLY the current file
tgt, out = load_sounds(path, mtime)
tgt_wav = to_wav_bytes(tgt, SAMPLE_RATE, key=f"{current}-tgt-{mtime}")
out_wav = to_wav_bytes(out, SAMPLE_RATE, key=f"{current}-out-{mtime}")

st.write("🔊 Target")
st.audio(tgt_wav, format="audio/wav")
st.write("🔊 Output")
st.audio(out_wav, format="audio/wav")

default_val = int(ratings.get(current, 3))
st.slider(
    "Similarity (1–5)",
    1, 5,
    value=default_val,
    step=1,
    key=f"score_{current}",
)

st.caption("Ratings are saved automatically when you press Prev/Next or change Program.")

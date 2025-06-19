import streamlit as st
import sounddevice as sd
import numpy as np
import librosa
import time
import threading
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from starlette.responses import JSONResponse

# Set up page config
st.set_page_config(
    page_title="Speech Analysis",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Remove whitespace
st.markdown("""
<style>
    .reportview-container .main .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Global variables
sr = 22050  # Sampling rate
running = False
analysis_thread = None

def extract_features(audio, sr):
    features = {}
    # Ensure mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # Extract features
    pitch, _ = librosa.piptrack(y=audio, sr=sr)
    pitches = pitch[pitch > 0]
    features["pitch_mean"] = np.mean(pitches) if len(pitches) > 0 else 0
    features["pitch_std"] = np.std(pitches) if len(pitches) > 0 else 0
    
    features["rms_mean"] = np.mean(librosa.feature.rms(y=audio))
    features["zcr_mean"] = np.mean(librosa.feature.zero_crossing_rate(y=audio))
    features["tempo"] = librosa.beat.tempo(y=audio, sr=sr)[0]
    features["mfccs"] = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13), axis=1)

    # Additional features for Clarity
    features["spectral_centroid"] = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    features["spectral_bandwidth"] = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
    features["chroma_mean"] = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr))

    return features

def analyze_speech(features):
    analysis = []

    # Pitch
    if features["pitch_mean"] < 100:
        analysis.append("⚠️ Voice may sound dull or low-pitched.")
    elif features["pitch_mean"] > 250:
        analysis.append("⚠️ Voice might be too high-pitched.")
    else:
        analysis.append("✅ Pitch is within a natural speaking range.")

    if features["pitch_std"] < 10:
        analysis.append("⚠️ Try adding more pitch variation for expressiveness.")
    else:
        analysis.append("✅ Good pitch variation detected.")

    # Loudness / Energy
    if features["rms_mean"] < 0.02:
        analysis.append("⚠️ Speaking too softly. Increase volume.")
    elif features["rms_mean"] > 0.1:
        analysis.append("⚠️ Voice might be too loud or harsh.")
    else:
        analysis.append("✅ Volume level seems appropriate.")

    # Tempo (words per minute approximation)
    if features["tempo"] < 90:
        analysis.append("⚠️ You may be speaking too slowly.")
    elif features["tempo"] > 160:
        analysis.append("⚠️ You may be speaking too fast.")
    else:
        analysis.append("✅ Speaking pace looks natural.")

    # Optional: ZCR can indicate articulation or sharpness
    if features["zcr_mean"] > 0.1:
        analysis.append("⚠️ Speech may be sharp or hissy (check articulation).")
    else:
        analysis.append("✅ Speech articulation is within a normal range.")
    
    clarity_warnings = []
    if features["spectral_centroid"] < 1500:
        clarity_warnings.append("🔈 Try to speak more clearly or with more energy.")

    if features["spectral_bandwidth"] < 1800:
        clarity_warnings.append("📉 Your voice may sound muffled or dull — increase enunciation.")

    if features["chroma_mean"] < 0.3:
        clarity_warnings.append("🎵 Add more variation to your pitch for expressive delivery.")
    
    return analysis + clarity_warnings

# Create containers for our data
header = st.container()
metrics = st.container()
analysis_container = st.container()

with header:
    st.markdown("<h3 style='text-align: center;'>Real-time Speech Analysis</h3>", unsafe_allow_html=True)

# Define session state to store analysis results
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = []
if 'metrics_data' not in st.session_state:
    st.session_state.metrics_data = {"pitch": 0, "energy": 0, "tempo": 0, "zcr": 0}

def audio_analysis_thread():
    global running
    duration = 3  # seconds per chunk
    
    while running:
        try:
            audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
            sd.wait()
            chunk = audio.flatten()

            features = extract_features(chunk, sr)
            analysis = analyze_speech(features)
            
            # Update session state
            st.session_state.metrics_data = {
                "pitch": f"{features['pitch_mean']:.1f} Hz",
                "energy": f"{features['rms_mean']:.3f}",
                "tempo": f"{features['tempo']:.1f} BPM",
                "zcr": f"{features['zcr_mean']:.3f}"
            }
            st.session_state.analysis_results = analysis
            
            time.sleep(0.2)  # Small buffer
        except Exception as e:
            st.error(f"Error in audio analysis: {e}")
            running = False
            break

# FastAPI app for handling start/stop requests
app = FastAPI()

# Add CORS middleware to allow requests from your HTML page
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify your exact origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/start_analysis")
async def start_analysis(request: Request):
    global running, analysis_thread
    if not running:
        running = True
        analysis_thread = threading.Thread(target=audio_analysis_thread)
        analysis_thread.start()
        return JSONResponse(content={"status": "Analysis started"})
    return JSONResponse(content={"status": "Analysis already running"})

@app.post("/stop_analysis")
async def stop_analysis(request: Request):
    global running
    if running:
        running = False
        if analysis_thread and analysis_thread.is_alive():
            analysis_thread.join(timeout=1)
        return JSONResponse(content={"status": "Analysis stopped"})
    return JSONResponse(content={"status": "Analysis already stopped"})

# Main Streamlit app displaying results
with metrics:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Pitch", st.session_state.metrics_data["pitch"])
    with col2:
        st.metric("Energy", st.session_state.metrics_data["energy"])
    with col3:
        st.metric("Tempo", st.session_state.metrics_data["tempo"])
    with col4:
        st.metric("Articulation", st.session_state.metrics_data["zcr"])

with analysis_container:
    st.markdown("### Analysis")
    for item in st.session_state.analysis_results:
        st.markdown(f"• {item}")
        
# Run FastAPI alongside Streamlit
def run_fastapi():
    uvicorn.run(app, host="127.0.0.1", port=9000)

if __name__ == "__main__":
    # Start FastAPI in a separate thread
    fastapi_thread = threading.Thread(target=run_fastapi)
    fastapi_thread.daemon = True
    fastapi_thread.start()
    
    # Keep updating the Streamlit UI
    while True:
        time.sleep(1)
        st.rerun()
        
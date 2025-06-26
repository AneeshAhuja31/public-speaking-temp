import streamlit as st
import sounddevice as sd
import numpy as np
import librosa
import pandas as pd
import time # To potentially add a small sleep for CPU management
import altair as alt
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Configuration ---
sr = 22050
duration = 3 # seconds per audio chunk

# --- Helper for Normalization ---
def normalize_feature(value, min_val, max_val):
    """Normalize a value to the range [0, 1] given min and max possible values."""
    if max_val == min_val: # Avoid division by zero, return midpoint
        return 0.5
    return (value - min_val) / (max_val - min_val)

# --- Define expected min/max for normalization (adjust as needed) ---
# These are rough estimates; you might fine-tune them based on typical vocal ranges.
MIN_PITCH = 50   # Hz (for very low voices, e.g., male fundamental)
MAX_PITCH = 400  # Hz (for relatively high voices, e.g., female fundamental)

MIN_RMS = 0.001  # A very quiet sound
MAX_RMS = 0.25   # A loud sound (can go higher but this keeps typical speech visible)

MIN_TEMPO = 60   # BPM (very slow speaking)
MAX_TEMPO = 200  # BPM (very fast speaking)

# --- Define ideal ranges for public speaking ---
IDEAL_PITCH_MIN = 100  # Hz - lower bound of ideal pitch range
IDEAL_PITCH_MAX = 250  # Hz - upper bound of ideal pitch range

IDEAL_RMS_MIN = 0.03   # lower bound of ideal loudness
IDEAL_RMS_MAX = 0.12   # upper bound of ideal loudness

IDEAL_TEMPO_MIN = 120  # BPM - lower bound of ideal speaking tempo
IDEAL_TEMPO_MAX = 150  # BPM - upper bound of ideal speaking tempo


# --- Feature Extraction Function ---
def extract_features(audio, sr):
    features = {}
    # Clean the audio buffer to prevent "not finite everywhere" errors
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Ensure audio is float32 for librosa operations if it's not already
    audio = audio.astype(np.float32)
    
    # Clip audio to reasonable range to prevent overflow
    audio = np.clip(audio, -1.0, 1.0)

    # Handle very short or silent audio chunks at the beginning
    if len(audio) < sr * 0.1 or np.all(np.abs(audio) < 1e-6): # If very short or near silent
        return {
            "pitch_mean": 0.0, "pitch_std": 0.0, "rms_mean": 0.0,
            "zcr_mean": 0.0, "tempo": 0.0, "mfccs": [0.0]*13,
            "spectral_centroid": 0.0, "spectral_bandwidth": 0.0, "chroma_mean": 0.0,
            "mfcc_plot_data": np.zeros((13,1)) # Placeholder for MFCC plot data
        }

    # Pitch tracking (using librosa.pyin for better accuracy for voice)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f0, voiced_flag, voiced_probabilities = librosa.pyin(
                y=audio, sr=sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C5'),
                frame_length=2048, # Increase frame_length for more stable pitch tracking on longer chunks
                hop_length=512 # Smaller hop_length for more detail
            )
            pitches = f0[~np.isnan(f0) & np.isfinite(f0)]
            features["pitch_mean"] = float(np.mean(pitches)) if len(pitches) > 0 else 0.0
            features["pitch_std"] = float(np.std(pitches)) if len(pitches) > 0 else 0.0
    except Exception: # Catch broader exceptions for robustness
        features["pitch_mean"] = 0.0
        features["pitch_std"] = 0.0
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rms_values = librosa.feature.rms(y=audio)
            features["rms_mean"] = float(np.mean(rms_values[np.isfinite(rms_values)]))
    except Exception:
        features["rms_mean"] = 0.0
        
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            zcr_values = librosa.feature.zero_crossing_rate(y=audio)
            features["zcr_mean"] = float(np.mean(zcr_values[np.isfinite(zcr_values)]))
    except Exception:
        features["zcr_mean"] = 0.0
        
    # Tempo (requires sufficient audio length)
    try:
        if len(audio) >= sr * 2: # At least 2 seconds for tempo estimation
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Use the updated path if available, fall back to the old one
                try:
                    from librosa.feature.rhythm import tempo
                    tempo_val = tempo(y=audio, sr=sr)[0]
                except ImportError:
                    tempo_val = librosa.beat.tempo(y=audio, sr=sr)[0]
                features["tempo"] = float(tempo_val) if np.isfinite(tempo_val) else 0.0
        else:
            features["tempo"] = 0.0
    except Exception:
        features["tempo"] = 0.0
        
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            mfcc_means = np.mean(mfccs, axis=1)
            mfcc_means = mfcc_means[np.isfinite(mfcc_means)]
            if len(mfcc_means) >= 13:
                features["mfccs"] = [float(val) for val in mfcc_means[:13]]
            else:
                features["mfccs"] = [0.0]*13
            features["mfcc_plot_data"] = mfccs # Store raw MFCCs for plotting
    except Exception:
        features["mfccs"] = [0.0]*13
        features["mfcc_plot_data"] = np.zeros((13,1)) # Placeholder for MFCC plot data

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sc_values = librosa.feature.spectral_centroid(y=audio, sr=sr)
            features["spectral_centroid"] = float(np.mean(sc_values[np.isfinite(sc_values)]))
    except Exception:
        features["spectral_centroid"] = 0.0
        
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sb_values = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
            features["spectral_bandwidth"] = float(np.mean(sb_values[np.isfinite(sb_values)]))
    except Exception:
        features["spectral_bandwidth"] = 0.0
        
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chroma_values = librosa.feature.chroma_stft(y=audio, sr=sr)
            features["chroma_mean"] = float(np.mean(chroma_values[np.isfinite(chroma_values)]))
    except Exception:
        features["chroma_mean"] = 0.0
        
    return features

# --- Speech Analysis (Alerts) Function ---
def analyze_speech(features):
    alerts = []

    # Helper function to safely get feature values
    def get_feature(feature_name, default_value=0.0):
        return features.get(feature_name, default_value)

    pitch_mean = get_feature("pitch_mean")
    pitch_std = get_feature("pitch_std")
    rms_mean = get_feature("rms_mean")
    tempo = get_feature("tempo")
    zcr_mean = get_feature("zcr_mean")
    spectral_centroid = get_feature("spectral_centroid")
    spectral_bandwidth = get_feature("spectral_bandwidth")
    chroma_mean = get_feature("chroma_mean")

    # 🎵 Pitch
    if pitch_mean > 0: # Only alert if pitch was detected
        # These thresholds are for typical speech (adult male/female)
        # Adjust as needed for specific target ranges
        if pitch_mean < 80: # Very low pitch (e.g., whispering or extremely deep voice)
            alerts.append("📢 Your pitch is quite low. Try speaking with more energy or varying your tone.")
        elif pitch_mean > 280: # Very high pitch
            alerts.append("📢 Your pitch is unusually high. Consider a more relaxed, natural tone.")

    if pitch_std > 0 and pitch_std < 13: # Low variation in pitch
        alerts.append("🎙️ Your voice lacks variation. Try adding some pitch dynamics for expressiveness.")

    # 🔊 Loudness / Energy
    if rms_mean < 0.015:
        alerts.append("🔈 You're speaking too softly. Increase your volume for better audibility.")
    elif rms_mean > 0.18:
        alerts.append("🔊 Your volume is quite high. Lower it slightly to avoid sounding aggressive.")

    # 🕒 Tempo
    if tempo > 0: # Only alert if tempo was estimated
        if tempo < 90:
            alerts.append("🐢 You may be speaking too slowly. Increase your pace to keep listeners engaged.")
        elif tempo > 160:
            alerts.append("⚡ You're speaking too fast. Try slowing down for better clarity and comprehension.")

    # 🫧 Articulation via ZCR (Zero Crossing Rate)
    # High ZCR can indicate noise, sibilance, or very rapid, crisp articulation
    if zcr_mean > 0.15:
        alerts.append("💨 Your speech has high sibilance or sharpness. Focus on precise articulation without hissing.")

    # 🧠 Spectral Centroid (Vocal Brightness/Muffledness)
    if spectral_centroid > 0 and spectral_centroid < 1500: # Lower values can indicate a muffled or dull sound
        alerts.append("🔈 Your voice might sound a bit muffled. Try to speak more clearly or with more frontal resonance.")

    # 📶 Spectral Bandwidth (Vocal Fullness/Resonance)
    if spectral_bandwidth > 0 and spectral_bandwidth < 1800: # Lower values can indicate a thin or less resonant voice
        alerts.append("📉 Your voice may sound dull or lack fullness — try increasing enunciation and opening your mouth more.")

    # 🎶 Chroma (Vocal Expressiveness/Musicality)
    if chroma_mean > 0 and chroma_mean < 0.3: # Lower values can indicate less harmonic richness or expressiveness
        alerts.append("🎵 Add more pitch variation for a dynamic voice. Let your voice rise and fall to convey emotion.")

    return alerts

# --- Function to create individual plots with shaded ideal ranges ---
def create_individual_plot(df, metric, title, y_label, ideal_min, ideal_max, color):
    """Create an individual plot with shaded ideal range"""
    if df.empty:
        return None
    
    # Create base chart with updated Altair syntax
    brush = alt.selection_interval(bind='scales')
    base = alt.Chart(df).add_params(brush)  # Updated from add_selection to add_params
    
    # Create a separate dataframe for the shaded region
    time_range = [df['Time (s)'].min(), df['Time (s)'].max()]
    shade_data = pd.DataFrame({
        'Time (s)': time_range,
        'ideal_min': [ideal_min, ideal_min],
        'ideal_max': [ideal_max, ideal_max]
    })
    
    # Create the shaded ideal range
    ideal_range = alt.Chart(shade_data).mark_area(
        opacity=0.3,
        color='lightgreen'
    ).encode(
        x=alt.X('Time (s):Q', title='Time (seconds)'),
        y='ideal_min:Q',
        y2='ideal_max:Q'
    )
    
    # Create the line chart
    line = base.mark_line(
        point=True,
        color=color,
        strokeWidth=2
    ).encode(
        x=alt.X('Time (s):Q', title='Time (seconds)'),
        y=alt.Y(f'{metric}:Q', title=y_label),
        tooltip=['Time (s):Q', f'{metric}:Q']
    )
    
    # Combine the charts
    chart = (ideal_range + line).resolve_scale(
        y='shared'
    ).properties(
        title=title,
        width=600,
        height=200
    )
    
    return chart

# --- Streamlit UI ---
st.title("🎙️ Live Pitch and Tone Analyzer")

st.info("Allow microphone access if prompted. Speak clearly into your microphone to get real-time feedback.")

placeholder = st.empty()

# Initialize session state for storing historical data for plotting
if 'feature_history' not in st.session_state:
    st.session_state.feature_history = pd.DataFrame(columns=['Time (s)', 'Pitch (Hz)', 'Loudness (RMS)', 'Tempo (BPM)'])
if 'current_time_s' not in st.session_state:
    st.session_state.current_time_s = 0

# Continuous loop - runs indefinitely
while True:
    try:
        # Record audio for `duration` seconds
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait() # Wait for the recording to finish

        chunk = audio.flatten()
        
        # Apply fix for non-finite values (NaNs, Infs) that might come from sounddevice
        if not np.isfinite(chunk).all():
            chunk = np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)

        features = extract_features(chunk, sr)
        alerts = analyze_speech(features)

        # Create new row data
        new_data = {
            'Time (s)': st.session_state.current_time_s,
            'Pitch (Hz)': features['pitch_mean'],
            'Loudness (RMS)': features['rms_mean'],
            'Tempo (BPM)': features['tempo']
        }
        
        # Fix for pandas concatenation warning - check if dataframe is empty first
        if st.session_state.feature_history.empty:
            st.session_state.feature_history = pd.DataFrame([new_data])
        else:
            new_row = pd.DataFrame([new_data])
            st.session_state.feature_history = pd.concat([st.session_state.feature_history, new_row], ignore_index=True)
        
        st.session_state.current_time_s += duration # Increment time for next chunk

        # Keep only the last N seconds of data for a moving window effect
        max_history_seconds = 30 # Display last 30 seconds of data
        st.session_state.feature_history = st.session_state.feature_history[
            st.session_state.feature_history['Time (s)'] >= (st.session_state.current_time_s - max_history_seconds)
        ]

        with placeholder.container():
            st.subheader("📊 Live Vocal Analysis")

            if not st.session_state.feature_history.empty:
                # PITCH SECTION
                st.markdown("### 🎵 Pitch Analysis")
                pitch_col1, pitch_col2 = st.columns([1, 3])
                with pitch_col1:
                    st.write(f"**Pitch (Mean):** {features['pitch_mean']:.1f} Hz")
                    st.write(f"**Pitch (Std Dev):** {features['pitch_std']:.1f} Hz")
                with pitch_col2:
                    pitch_chart = create_individual_plot(
                        st.session_state.feature_history, 
                        'Pitch (Hz)', 
                        '🎵 Pitch Analysis',
                        'Pitch (Hz)', 
                        IDEAL_PITCH_MIN, 
                        IDEAL_PITCH_MAX, 
                        '#1f77b4'
                    )
                    if pitch_chart:
                        st.altair_chart(pitch_chart, use_container_width=True)
                        st.caption("Green shaded area represents ideal pitch range for public speaking (100-250 Hz)")

                st.markdown("---")

                # LOUDNESS SECTION
                st.markdown("### 🔊 Loudness Analysis")
                loud_col1, loud_col2 = st.columns([1, 3])
                with loud_col1:
                    st.write(f"**Loudness (RMS):** {features['rms_mean']:.3f}")
                with loud_col2:
                    loudness_chart = create_individual_plot(
                        st.session_state.feature_history, 
                        'Loudness (RMS)', 
                        '🔊 Loudness Analysis',
                        'Loudness (RMS)', 
                        IDEAL_RMS_MIN, 
                        IDEAL_RMS_MAX, 
                        '#ff7f0e'
                    )
                    if loudness_chart:
                        st.altair_chart(loudness_chart, use_container_width=True)
                        st.caption("Green shaded area represents ideal loudness range for public speaking (0.03-0.12 RMS)")

                st.markdown("---")

                # TEMPO SECTION
                st.markdown("### ⏱️ Tempo Analysis")
                tempo_col1, tempo_col2 = st.columns([1, 3])
                with tempo_col1:
                    st.write(f"**Tempo:** {features['tempo']:.1f} BPM")
                with tempo_col2:
                    tempo_chart = create_individual_plot(
                        st.session_state.feature_history, 
                        'Tempo (BPM)', 
                        '⏱️ Tempo Analysis',
                        'Tempo (BPM)', 
                        IDEAL_TEMPO_MIN, 
                        IDEAL_TEMPO_MAX, 
                        '#2ca02c'
                    )
                    if tempo_chart:
                        st.altair_chart(tempo_chart, use_container_width=True)
                        st.caption("Green shaded area represents ideal speaking tempo for public speaking (120-150 BPM)")

                st.markdown("---")

            else:
                st.info("Start speaking to see vocal metrics over time!") 

            # ALERTS SECTION — at the end, below everything
            st.subheader("🗣️ Vocal Feedback")
            if alerts:
                for item in alerts:
                    st.warning(item)
            else:
                st.success("✨ Your vocal delivery sounds great in this segment!")
    
        
        # Add a small sleep to prevent the loop from consuming too much CPU unnecessarily
        # This is particularly important for Streamlit's reruns
        time.sleep(0.1) # You can uncomment and adjust this if UI becomes unresponsive, though sd.wait() already blocks

    except Exception as e:
        # If there's an error, display it in the Streamlit app and continue
        with placeholder.container():
            st.error(f"Error in processing audio: {e}")
            st.info("Please ensure your microphone is connected and accessible. Restarting analysis...")
        # A small pause after an error before retrying to prevent rapid error looping
        time.sleep(1) # Uncomment if errors are causing very rapid UI flickering
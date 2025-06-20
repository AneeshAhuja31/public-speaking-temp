import streamlit as st
import sounddevice as sd
import numpy as np
import librosa

sr = 22050
duration = 3

def extract_features(audio, sr):
    features = {}
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    pitch, _ = librosa.piptrack(y=audio, sr=sr)
    pitches = pitch[pitch > 0]
    features["pitch_mean"] = np.mean(pitches) if len(pitches) > 0 else 0
    features["pitch_std"] = np.std(pitches) if len(pitches) > 0 else 0
    features["rms_mean"] = np.mean(librosa.feature.rms(y=audio))
    features["zcr_mean"] = np.mean(librosa.feature.zero_crossing_rate(y=audio))
    features["tempo"] = librosa.beat.tempo(y=audio, sr=sr)[0]
    features["mfccs"] = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13), axis=1)
    features["spectral_centroid"] = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    features["spectral_bandwidth"] = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
    features["chroma_mean"] = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr))
    return features

def analyze_speech(features):
    analysis = []

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
    if features["rms_mean"] < 0.02:
        analysis.append("⚠️ Speaking too softly. Increase volume.")
    elif features["rms_mean"] > 0.1:
        analysis.append("⚠️ Voice might be too loud or harsh.")
    else:
        analysis.append("✅ Volume level seems appropriate.")
    if features["tempo"] < 90:
        analysis.append("⚠️ You may be speaking too slowly.")
    elif features["tempo"] > 160:
        analysis.append("⚠️ You may be speaking too fast.")
    else:
        analysis.append("✅ Speaking pace looks natural.")
    if features["zcr_mean"] > 0.1:
        analysis.append("⚠️ Speech may be sharp or hissy (check articulation).")
    else:
        analysis.append("✅ Speech articulation is within a normal range.")
    if features["spectral_centroid"] < 1500:
        analysis.append("🔈 Try to speak more clearly or with more energy.")
    if features["spectral_bandwidth"] < 1800:
        analysis.append("📉 Your voice may sound muffled or dull — increase enunciation.")
    if features["chroma_mean"] < 0.3:
        analysis.append("🎵 Add more variation to your pitch for expressive delivery.")
    return analysis

st.title("🎙️ Live Pitch/Tone Analyzer")

placeholder = st.empty()

while True:
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    chunk = audio.flatten()

    features = extract_features(chunk, sr)
    analysis = analyze_speech(features)

    with placeholder.container():
        st.subheader("Features")
        st.write(f"Pitch: {features['pitch_mean']:.1f} Hz")
        st.write(f"RMS: {features['rms_mean']:.3f}")
        st.write(f"Tempo: {features['tempo']:.1f} BPM")

        st.subheader("Analysis")
        for item in analysis:
            st.write("•", item)

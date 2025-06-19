import sounddevice as sd
import numpy as np
import librosa
import time

sr = 22050  # Sampling rate

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
    
    if features["spectral_centroid"] < 1500:
        print("🔈 Try to speak more clearly or with more energy.")

    if features["spectral_bandwidth"] < 1800:
        print("📉 Your voice may sound muffled or dull — increase enunciation.")

    if features["chroma_mean"] < 0.3:
        print("🎵 Add more variation to your pitch for expressive delivery.")


    return analysis

duration = 3  # seconds per chunk
print("🎤 Listening... (Ctrl+C to stop)")

try:
    while True:
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
        chunk = audio.flatten()

        features = extract_features(chunk, sr)
        
        analysis= analyze_speech(features)
        # Display (replace with streamlit/web socket later)
        print(f"🎧 PITCH: {features['pitch_mean']:.1f} Hz | ENERGY: {features['rms_mean']:.3f} | TEMPO: {features['tempo']:.1f} BPM")
        print(f"ZCR: {features['zcr_mean']:.3f}")
        print("MFCCs:", ", ".join([f"{v:.2f}" for v in features["mfccs"][:5]]), "...")
        print("Analysis:")
        for a in analysis:
            print("•", a)
        print("------")
        
        time.sleep(0.2)  # Optional: slight buffer

except KeyboardInterrupt:
    print("🛑 Stopped.")


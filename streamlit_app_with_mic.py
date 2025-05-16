# import streamlit as st
# from audiorecorder import audiorecorder
# import tempfile
# import requests

# st.title("🎙️ Whisper Transcription Demo")

# # Audio recording
# audio = audiorecorder("🎙️ Click to Record", "Recording...")

# if len(audio) > 0:
#     # Save to WAV using export
#     with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
#         audio.export(f, format="wav")
#         temp_wav_path = f.name

#     # Playback in Streamlit
#     with open(temp_wav_path, "rb") as audio_file:
#         st.audio(audio_file.read(), format="audio/wav")
#         audio_file.seek(0)  # rewind

#         # Send to Whisper backend
#         print("Sending audio to Whisper backend...", audio_file)
#         res = requests.post("http://localhost:5001/transcribe", files={"audio": audio_file})
#         transcription = res.json().get("text", "")
#     # Show transcription inside input
#     prompt = st.text_input("Transcription:", value=transcription)
# else:
#     st.info("Click mic and speak. It'll transcribe and fill the input box.")


import streamlit as st
import sounddevice as sd
import numpy as np
import tempfile
import scipy.io.wavfile as wav
import time
import os
import requests
import io

st.set_page_config(page_title="Mic Recorder")

# Session setup
if "recording" not in st.session_state:
    st.session_state.recording = False

# UI layout
col1, col2 = st.columns([10, 1])
with col1:
    query_text = st.text_input("Ask a question about your inventory:", value=st.session_state.get("query_input", ""), key="query_input", label_visibility="collapsed")
with col2:
    if st.button("🎤", key="mic_button"):
        st.session_state.recording = True
        st.experimental_rerun()

# Recorder function with silence detection
def record_until_silence(threshold_db=-35, silence_sec=3, max_duration=30, fs=16000):
    st.info("Recording... Speak into the mic.")

    chunk = int(fs * 0.5)  # half-second chunks
    silence_chunks = 0
    audio = []

    start_time = time.time()
    for _ in range(int(max_duration * 2)):  # 0.5s chunks
        data = sd.rec(chunk, samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        audio.append(data.copy())

        # Convert to dB
        volume = 20 * np.log10(np.max(np.abs(data)) + 1)
        if volume < threshold_db:
            silence_chunks += 1
        else:
            silence_chunks = 0

        # Stop after 3s of silence
        if silence_chunks * 0.5 >= silence_sec:
            break

    full_audio = np.concatenate(audio)
    return fs, full_audio

# If recording is triggered
if st.session_state.get("recording", False):
    fs, recorded_audio = record_until_silence()

    # Save to WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        wav.write(tmpfile.name, fs, recorded_audio)
        audio_path = tmpfile.name

    # Playback
    st.audio(audio_path, format="audio/wav")

    # Send to Whisper
    st.success("Transcribing...")
    with open(audio_path, "rb") as f:
        try:
            res = requests.post("http://localhost:5001/transcribe", files={"audio": f})
            res.raise_for_status()
            transcription = res.json().get("text", "")
            if transcription:
                st.session_state.transcribed_text = transcription
                st.session_state.transcription_ready = True
                st.session_state.recording = False
                st.rerun()
            else:
                st.warning("No transcription returned.")
        except Exception as e:
            st.error(f"Transcription failed: {e}")

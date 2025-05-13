import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, ClientSettings
import av
import openai

# Set your OpenAI key securely
openai.api_key = st.secrets["OPENAI_API_KEY"]

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.frames.append(frame.to_ndarray().tobytes())
        return frame

st.title("🎙️ Record and Transcribe Audio")

ctx = webrtc_streamer(
    key="example",
    mode="sendonly",
    in_audio=True,
    audio_processor_factory=AudioProcessor,
    client_settings=ClientSettings(
        media_stream_constraints={"audio": True, "video": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    ),
)

if st.button("Transcribe"):
    if ctx.audio_processor and ctx.audio_processor.frames:
        audio_data = b"".join(ctx.audio_processor.frames)

        # Save to file
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_data)

        # Transcribe with Whisper
        with open("temp_audio.wav", "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)

        st.markdown("### 📝 Transcription:")
        st.write(transcript["text"])
    else:
        st.warning("No audio data to transcribe.")

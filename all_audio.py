import streamlit as st
import pandas as pd
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, ClientSettings
import openai
import av

# Load OpenAI API key securely
openai.api_key = "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"

st.title("🔊📊 Voice-Powered CSV Analyzer")

# CSV upload section
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

# Audio recording processor
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.frames.append(frame.to_ndarray().tobytes())
        return frame

# Streamlit WebRTC component
ctx = webrtc_streamer(
    key="voice-analyzer",
    mode="sendonly",
    in_audio=True,
    audio_processor_factory=AudioProcessor,
    client_settings=ClientSettings(
        media_stream_constraints={"audio": True, "video": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    ),
)

# Transcription button
if st.button("🎧 Transcribe & Analyze"):
    if ctx.audio_processor and ctx.audio_processor.frames:
        audio_data = b"".join(ctx.audio_processor.frames)

        with open("temp_audio.wav", "wb") as f:
            f.write(audio_data)

        with open("temp_audio.wav", "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)

        question = transcript["text"]
        st.markdown(f"### 🗣️ Transcribed Question: `{question}`")

        if df is not None:
            prompt = f"""
You are a helpful data analyst for my diamonds inventory.

Sample data:
{df.head(10).to_string()}

User question: {question}

Please analyze the dataset and answer the question clearly.
"""
            with st.spinner("Analyzing with GPT..."):
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a data analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )

            answer = response.choices[0].message.content
            st.subheader("🔍 GPT Analysis")
            st.markdown(answer)
        else:
            st.warning("Please upload a CSV file to analyze.")
    else:
        st.warning("No audio data captured yet.")

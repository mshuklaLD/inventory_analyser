import streamlit as st
import pandas as pd
from openai import OpenAI
from gtts import gTTS
import base64
import tempfile
import os
from streamlit_mic_recorder import mic_recorder

# 📌 Title
st.set_page_config(page_title="Inventory Intelligence with Audio", layout="wide")
st.title("📊 Inventory Intelligence with Voice")

# 🔑 OpenAI Client
client = OpenAI(api_key="your-api-key-here")

# 📂 Upload CSV
df = None
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

# 🗣️ Audio Recorder (front-end mic)
st.markdown("## 🎤 Ask a question by voice")
audio = mic_recorder(start_prompt="Start Recording", stop_prompt="Stop", just_once=True, use_container_width=True, key="rec")

if audio and "text" in audio:
    question = audio["text"]
    st.success(f"You asked: {question}")

    # Sample prompt with small data sample
    column_descriptions = """
    - Aging: How long the stone has been in my inventory, measured in days.
    - Rank: How does my stone rank compared to other similar stones on the RapNet platform based on the price per carat. Rank 1 is the best.
    - Count: Total number of similar stones available on the RapNet platform.
    - My Stock: Number of stones similar to this that I have in my inventory. If it says 1, it means there are no other stones like this in my inventory.
    - My Sales: Number of similar items I have sold in the last 2 months.
    - My YTD Sales: Number of similar items I have sold since the beginning of 2025.
    - Rap $/Ct: Price per carat of the diamond according to the Rapaport price list.
    - Rap Total: Total price of the diamond according to the Rapaport price list.
    - Rap %: Discount percentage of the Rapaport price that I am asking for this stone. For example, -40 means I'm asking 40% less than Rapaport.
    - $/Ct: Asking price per carat.
    - Total Price: Total asking price (Carat × $/Ct).
    """

    sample = df.head(10).to_string() if df is not None else ""
    summary = df.describe(include='all').to_string() if df is not None else ""

    prompt = f"""
    You are a helpful data analyst for my diamonds inventory.

    Dataset column descriptions:
    {column_descriptions}

    Dataset Sample:
    {sample}

    Summary Statistics:
    {summary}

    User Question: {question}
    """

    # GPT Analysis
    if df is not None:
        with st.spinner("Analyzing with GPT..."):
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful data analyst."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1000,
            )
            answer = response.choices[0].message.content
            st.markdown("### 🔍 GPT Answer")
            st.write(answer)

            # 🎧 Convert GPT answer to speech
            tts = gTTS(text=answer)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                tts.save(f.name)
                audio_path = f.name

            # Embed audio in Streamlit
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
                b64 = base64.b64encode(audio_bytes).decode()
                audio_html = f"""
                <audio controls autoplay>
                    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
                """
                st.markdown(audio_html, unsafe_allow_html=True)

            os.remove(audio_path)
    else:
        st.warning("Please upload a CSV file first.")

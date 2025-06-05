import streamlit as st
from elasticsearch import Elasticsearch
import json
from openai import OpenAI
import pandas as pd
from pydub import AudioSegment
import tempfile
import os
from audiorecorder import audiorecorder
import io
import requests
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# 🧠 Set your OpenAI API key (securely with secrets or env var in real apps)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="Liquid Diamonds Inventory Assistant", layout="centered")
# Add logo and title
st.markdown(
    """
    <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 1rem;">
        <img src="https://analytics.liquid.diamonds/assets/img/LD_logo_banner_long_dark_bg.png" alt="Liquid Diamonds Logo" width="400">
    </div>
    <h1 style="text-align: center; ">Pricing Co-Pilot</h1>
    <h5 style="text-align: center; ">Ask a question about your diamond inventory or use the mic below.</h5>
    """,
    unsafe_allow_html=True
)

# Deepseek API key
# client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com/v1")
# whisper_client = OpenAI(api_key=os.getenv("WHISPER_API_KEY"))

# Connect to your hosted Elasticsearch
# es = Elasticsearch("http://127.0.0.1:9200")
es = Elasticsearch("http://34.41.92.221:9200")    
# Check if the connection is successful
if not es.ping(): 
    st.error("Elasticsearch connection failed.")

column_descriptions = """
Here is a description of key columns in the dataset:

- Aging: How long the stone has been in my inventory, measured in days.
- Rank: How does my stone rank compared to other similar stones on the RapNet platform based on the price per carat. Rank 1 is the best.
- Count: Total number of similar stones available on the RapNet platform.
- My Stock: Number of stones similar to this that I have in my inventory.
- My Sales: Number of similar items I have sold in the last 2 months.
- My YTD Sales: Number of similar items I have sold since the beginning of 2025.
- Rap $/Ct: Price per carat of the diamond according to the Rapaport price list.
- Rap Total: Total price of the diamond according to the Rapaport price list.
- Rap %: Discount percentage of the Rapaport price that I am asking for this stone. For example, -40 means I'm asking 40% less than Rapaport.
- $/Ct: Asking price per carat.
- Total Price: Total asking price (Carat × $/Ct).
"""

def summarize_agg_to_text(agg_result: dict) -> str:
    summaries = []

    for agg_name, agg_data in agg_result.items():
        if "value" in agg_data:
            summaries.append(f"{agg_name.replace('_', ' ').capitalize()}: {int(agg_data['value'])}")
        elif "buckets" in agg_data:
            bucket_summaries = []
            for bucket in agg_data["buckets"]:
                desc = f"{bucket.get('key')}"
                if "doc_count" in bucket:
                    desc += f" ({bucket['doc_count']} docs)"
                for k, v in bucket.items():
                    if isinstance(v, dict) and "value" in v:
                        desc += f", {k.replace('_', ' ')}: {int(v['value'])}"
                bucket_summaries.append(desc)
            summaries.append(f"{agg_name.replace('_', ' ').capitalize()}: " + "; ".join(bucket_summaries))

    return "Summary: " + " | ".join(summaries)

def summarize_elasticsearch_response(results):
    if "aggregations" in results:
        st.subheader("Aggregation Summary")
        for agg_name, agg_data in results["aggregations"].items():
            if "value" in agg_data:
                st.write(f"→ {agg_name}: {agg_data['value']}")
            elif "buckets" in agg_data:
                st.write(f"→ {agg_name}:")
                for bucket in agg_data["buckets"]:
                    st.write({
                        "key": bucket.get("key"),
                        "doc_count": bucket.get("doc_count"),
                        **{k: v["value"] for k, v in bucket.items() if isinstance(v, dict) and "value" in v}
                    })
    else:
        hits = results.get("hits", {}).get("hits", [])
        total = results.get("hits", {}).get("total", {}).get("value", 0)
        st.subheader(f"Document Results ({total} hits)")
        for hit in hits:
            st.write(hit["_source"])

# Index and schema context
schema = {
    "index": "inventory_nl",  # Use the existing index name
    "fields": {
        "item_id": "long",
        "aging": "integer",
        "cert_number": "keyword",
        "stock_num": "keyword",
        "shape": "keyword",
        "carat": "float",
        "carat_range": "keyword",
        "color": "keyword",
        "clarity": "keyword",
        "cut": "keyword",
        "total_price": "double",
        "my_stock": "integer",
        "my_sales": "integer",
        "my_ytd_sales": "integer",
        "rap_pct": "float",
        "rank": "integer",
        "count": "integer",
        "category": "text",
        "lab": "keyword",
        "fluor": "keyword",
        "polish": "keyword",
        "sym": "keyword",
        "shade": "keyword",
        "milky": "keyword"
    }
}
if "query_input" not in st.session_state:
    st.session_state.query_input = ""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Handle rerun triggered by audio transcription before rendering input
if "transcription_ready" in st.session_state and st.session_state.transcription_ready:
    # if not st.session_state.get("query_input"):  # Only update if empty
    st.session_state.query_input = st.session_state.transcribed_text
    st.session_state.transcription_ready = False  # reset flag
    st.rerun()

st.markdown("""
    <style>
    .input-row {
        display: flex;
        align-items: center;
        margin-top: 2em;
    }
    .input-box {
        flex-grow: 1;
    }
    .mic-button {
        background-color: #f0f2f6;
        border: none;
        padding: 0.5rem 0.8rem;
        margin-left: 0.5rem;
        border-radius: 10px;
        font-size: 1.5rem;
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="input-row">', unsafe_allow_html=True)

col1, col2 = st.columns([10, 1])
with col1:
    query_text = st.text_input("Ask a question about your inventory:", value=st.session_state.query_input, key="query_input", label_visibility="collapsed")

with col2:
    audio = audiorecorder("🎙️", "🔴", key="recorder")
    if len(audio) > 0:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio.export(f, format="wav")
            temp_wav_path = f.name
        # st.success("Transcribing...")

        # TODO: Replace this with your Whisper API or local transcription
        # dummy transcript:
        # Playback in Streamlit
        with open(temp_wav_path, "rb") as audio_file:
            # st.audio(audio_file.read(), format="audio/wav")
            audio_file.seek(0)  # rewind

            # Send to Whisper backend
            print("Sending audio to Whisper backend...", audio_file)
            try:
                res = requests.post("http://34.58.46.124:5001/transcribe", files={"audio": audio_file})
                res.raise_for_status()
                transcription = res.json().get("text", "")
                if not transcription:
                    st.warning("No transcription text returned.")
                else:
                    st.session_state.transcribed_text = transcription
                    st.session_state.transcription_ready = True
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to transcribe audio: {e}")
        # Show transcription inside input
        prompt = st.text_input("Transcription:", value=transcription)
        
st.markdown('</div>', unsafe_allow_html=True)
 
# If the user has provided a query, process it
if query_text:
    # Prompt OpenAI to translate NL → Elasticsearch
    prompt = f"""You are an Elasticsearch expert. Convert the user's natural language request into an Elasticsearch DSL query. Use this index schema: {json.dumps(schema, indent=2)}

User query: "{query_text}"

Use the following descriptions to understand the meaning of each column: "{column_descriptions}"

Just return the JSON query body only. Do NOT include "index" or "body" keys. Do not explain anything.
"""

    with st.spinner("Looking for results..."):
        # Updated for OpenAI >= 1.0.0
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are an Elasticsearch expert."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0
            # stream=False
        )

        # Print the raw response
        # st.write("Raw response from OpenAI:")
        # st.write(response)

        query_json = response.choices[0].message.content.strip()
        summary_text = ""  # NEW — collect summary to store in history

        try:
            parsed_query = json.loads(query_json)
            # st.subheader("Elasticsearch Query")
            # st.code(json.dumps(parsed_query, indent=2))

            # Run the query on the existing Elasticsearch index
            # if "index" in parsed_query:
            #     del parsed_query["index"]
            results = es.search(index=schema["index"], body=parsed_query)

            # Handle both hits and aggregations
            if "aggregations" in results:
                # st.subheader("Aggregations")
                # st.json(results["aggregations"])
                # Add natural language summary for aggregations
                # summarize_elasticsearch_response(results)
                # summarize_agg_to_text(results)
                summary_text = f"Aggregation result for {', '.join(results['aggregations'].keys())}"

                # Show aggregation results as tables
                for agg_name, agg_data in results["aggregations"].items():
                    if "buckets" in agg_data:
                        # Flatten bucket data
                        bucket_rows = []
                        for bucket in agg_data["buckets"]:
                            row = {"key": bucket.get("key"), "doc_count": bucket.get("doc_count")}
                            for k, v in bucket.items():
                                if isinstance(v, dict) and "value" in v:
                                    row[k] = v["value"]
                            bucket_rows.append(row)

                        agg_df = pd.DataFrame(bucket_rows)
                        st.subheader(f"{agg_name.replace('_', ' ').title()}")
                        # st.subheader(f"Aggregation: {agg_name}")
                        st.data_editor(agg_df)

                    elif "value" in agg_data:
                        # Single-value aggregation (e.g., avg, sum)
                        value_df = pd.DataFrame([{"metric": agg_name, "value": agg_data["value"]}])
                        st.subheader(f"{agg_name.replace('_', ' ').title()}")
                        # st.subheader(f"Aggregation: {agg_name}")
                        st.data_editor(value_df)
            elif "hits" in results and results["hits"]["hits"]:
                # st.subheader("Results")
                # st.json(results["hits"]["hits"])
                # Summarize hits in natural language
                hits = results["hits"]["hits"]
                total = results.get("hits", {}).get("total", {}).get("value", 0)
                summary_lines = []
                for hit in hits[:5]:  # summarize top 5 hits
                    source = hit["_source"]
                    summary = (
                        f"{source.get('shape', 'Unknown shape')} of {source.get('carat', '?')} ct, "
                        f"color {source.get('color', '?')}, clarity {source.get('clarity', '?')}, "
                        f"priced at ${source.get('total_price', '?')} total "
                        f"(Aging: {source.get('aging', '?')} days, Rank: {source.get('rank', '?')})"
                    )
                    summary_lines.append(summary)

                # st.subheader(f"Natural Language Summary ({total} hits)")
                st.write("Top matches:\n- " + "\n- ".join(summary_lines)) 
                summary_text = "Top matches:\n- " + "\n- ".join(summary_lines)
                if hits:
                    # Convert hits to DataFrame
                    records = [hit["_source"] for hit in hits]
                    df = pd.DataFrame(records)

                    # Optional: choose columns to display in the table
                    selected_cols = [
                        "shape", "carat", "color", "clarity", "cut", 
                        "total_price", "rap_pct", "rank", "my_stock", "my_sales"
                    ]
                    # Keep only available columns
                    selected_cols = [col for col in selected_cols if col in df.columns]
                    st.subheader("Results Table")
                    st.data_editor(df[selected_cols])
            else:
                st.info("No results found.")
                summary_text = "No results found."
            # Save query and summary to chat history
            st.session_state.chat_history.append({
                "query": query_text,
                "response": summary_text,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        except Exception as e:
            st.error(f"Error parsing or running query: {e}")

# Don't show chat history or any label if no query has been made
if st.session_state.chat_history:
    st.markdown(
        """
        <style>
        .chat-history {
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid #ccc;
            padding: 10px;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    for entry in st.session_state.chat_history[::-1]:  # reverse to show latest first
        timestamp = entry.get("timestamp", "🕒 Unknown time")
        user_message = entry.get("query", "❓ No query")
        assistant_response = entry.get("response", "❓ No response")
        with st.chat_message("user"):
            st.markdown(f"🕒 *{timestamp}*\n\n**You:** {user_message}")
        with st.chat_message("assistant"):
            st.markdown(f"**AI:** {assistant_response}")

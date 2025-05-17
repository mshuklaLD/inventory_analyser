import streamlit as st
from elasticsearch import Elasticsearch
import json
from openai import OpenAI
import requests
from dotenv import load_dotenv
load_dotenv()
import os
# 🧠 Set your OpenAI API key (securely with secrets or env var in real apps)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Connect to your hosted Elasticsearch
es = Elasticsearch("http://localhost:9200")
# Check if the connection is successful
if not es.ping(): 
    st.error("Elasticsearch connection failed.")

st.title("Natural Language Elasticsearch Search")

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

# Index and schema context
schema = {
    "index": "inventory_nl",  # Use the existing index name
    "fields": {
        "item_id": "long",
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
        "category": "text",
        "lab": "keyword",
        "fluor": "keyword",
        "polish": "keyword",
        "sym": "keyword",
        "shade": "keyword",
        "milky": "keyword"
    }
}

query_text = st.text_input("Ask a search question:", "")

if query_text:
    # Prompt OpenAI to translate NL → Elasticsearch
    prompt = f"""You are an Elasticsearch expert. Convert the user's natural language request into an Elasticsearch DSL query. Use this index schema: {json.dumps(schema, indent=2)}

User query: "{query_text}"

Use the following descriptions to understand the meaning of each column: "{column_descriptions}"

Just return the JSON query body only. Do NOT include "index" or "body" keys. Do not explain anything.
"""

    with st.spinner("Generating query..."):
        # Updated for OpenAI >= 1.0.0

        # response = requests.post(
        #     "https://api.aimlapi.com/v1/chat/completions",
        #     headers={"Authorization":"Bearer 8fb1229d497744b3a32484e21b062d2f","Content-Type":"application/json"},
        #     json={
        #         "model":"gpt-4o-mini",
        #         "frequency_penalty":1,
        #         "logit_bias":{"ANY_ADDITIONAL_PROPERTY":1},
        #         "logprobs":True,
        #         "top_logprobs":1,
        #         "max_tokens":512,
        #         "max_completion_tokens":512,
        #         "n":1,
        #         "prediction":{"type":"content","content":"text"},
        #         "presence_penalty":1,
        #         "seed":1,
        #         "messages":[{"role":"system","content":"text","name":"text"}],
        #         "stream":False,
        #         "stream_options":{"include_usage":True},
        #         "top_p":1,
        #         "temperature":1,
        #         "stop":"text",
        #         "tools":[
        #             {
        #                 "type":"function",
        #                 "function":{"description":"text","name":"text","parameters":None,"strict":True,"required":["text"]}
        #             }
        #         ],
        #         "tool_choice":"none",
        #         "parallel_tool_calls":True,
        #         "reasoning_effort":"low",
        #         "response_format":{"type":"text"},
        #         "audio":{"format":"wav","voice":"alloy"},
        #         "modalities":["text"],
        #         "web_search_options":{"search_context_size":"low","user_location":{"approximate":{"city":"text","country":"text","region":"text","timezone":"text"},"type":"approximate"}}
        #     }
        # )
        # response = requests.post(
        #     json={
        #         "model":"gpt-4o-mini",
        #         "messages":[
        #             {
        #                 "role":"user",

        #                 # Insert your question for the model here, instead of Hello:
        #                 "content":"Hello"
        #             }
        #         ]
        #     }
        # )

        # data = response.json()

        # data = response.json()

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an Elasticsearch expert."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        # Print the raw response
        # st.write("Raw response from OpenAI:")
        # st.write(response)

        query_json = response.choices[0].message.content.strip()

        try:
            parsed_query = json.loads(query_json)
            st.subheader("Elasticsearch Query")
            st.code(json.dumps(parsed_query, indent=2))

            # Run the query on the existing Elasticsearch index
            # if "index" in parsed_query:
            #     del parsed_query["index"]
            results = es.search(index=schema["index"], body=parsed_query)

            # Handle both hits and aggregations
            if "aggregations" in results:
                st.subheader("Aggregations")
                st.json(results["aggregations"])
            elif "hits" in results and results["hits"]["hits"]:
                st.subheader("Results")
                st.json(results["hits"]["hits"])
            else:
                st.info("No results found.")

        except Exception as e:
            st.error(f"Error parsing or running query: {e}")

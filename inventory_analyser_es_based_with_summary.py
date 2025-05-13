import streamlit as st
import openai
from elasticsearch import Elasticsearch
import json
from openai import OpenAI

# 🧠 Set your OpenAI API key (securely with secrets or env var in real apps)
client = OpenAI(
  api_key="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"
)

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
                # Add natural language summary for aggregations
                summarize_elasticsearch_response(results["aggregations"])
                summarize_agg_to_text(results["aggregations"])

            elif "hits" in results and results["hits"]["hits"]:
                st.subheader("Results")
                st.json(results["hits"]["hits"])
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

                st.subheader(f"Natural Language Summary ({total} hits)")
                st.write("Top matches:\n- " + "\n- ".join(summary_lines)) 
            else:
                st.info("No results found.")

        except Exception as e:
            st.error(f"Error parsing or running query: {e}")

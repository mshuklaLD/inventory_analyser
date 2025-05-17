import pandas as pd
import streamlit as st
from openai import OpenAI
import re
from dotenv import load_dotenv
load_dotenv()
import os
# 🧠 Set your OpenAI API key (securely with secrets or env var in real apps)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("📊 CSV Analyzer with OpenAI GPT")

# Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

# Mapping for keyword-driven summaries
keyword_map = {
    "slowest moving": {"metric": "Aging", "agg": "mean", "order": "desc"},
    "fastest moving": {"metric": "Aging", "agg": "mean", "order": "asc"},
    "highest sales": {"metric": "My Sales", "agg": "sum", "order": "desc"},
    "least sales": {"metric": "My Sales", "agg": "sum", "order": "asc"},
    "surplus": {"metric": "My Stock", "agg": "sum", "order": "desc"},
}

# Column descriptions for GPT context
column_descriptions = """
Here is a description of key columns in the dataset:
- Item ID: Unique identifier for each diamond in my inventory.
- Category: Define a skew for diamonds based on their characteristics.
- Cert: The certificate number of the diamond.
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

# Extract specific row based on Item ID
def extract_row_by_item_id(df, question):
    match = re.search(r'item id[^\d]*(\d+)', question.lower())
    if match and "item_id" in df.columns:
        item_id = match.group(1)
        df["item_id"] = df["item_id"].astype(str)
        row = df[df["item_id"] == item_id]
        if not row.empty:
            return row.to_string(index=False)
    return ""

# Detect if question matches known patterns
def detect_analysis_intent(user_question):
    for key, config in keyword_map.items():
        if key in user_question.lower():
            return config
    return None

# Generate summary based on detected intent
def generate_summary_from_intent(df, config, group_by="Category"):
    metric = config["metric"]
    agg_func = config["agg"]
    order = config["order"]
    if group_by not in df.columns or metric not in df.columns:
        return "Required column not found in data."
    grouped = df.groupby(group_by)[metric]
    summary = getattr(grouped, agg_func)().sort_values(ascending=(order == "asc")).head(10)
    return summary.to_string()

# Main execution
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    st.write("📄 **Sample Data**")
    st.dataframe(df.head(10))

    # Summary + Sample
    summary = df.describe(include='all').to_string()
    sample = df.head(10).to_string()
    st.write("📊 **Summary Statistics**")
    st.text(summary)

    st.write("🔎 Unique Item IDs in data (first 5):", df["item_id"].astype(str).unique()[:5])

    # Text Input
    question = st.text_input("Ask a question about the dataset (optional):")

    # Initialize session state for Q&A history
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    # Display history
    if st.session_state.qa_history:
        st.subheader("🕓 Chat History")
        for i, (q, a) in enumerate(reversed(st.session_state.qa_history[-5:])):  # show last 5
            st.markdown(f"**Q{i+1}:** {q}")
            st.markdown(f"**A{i+1}:** {a}")
    # Answer logic
    matched_row = extract_row_by_item_id(df, question)
    intent_config = detect_analysis_intent(question)
    dynamic_summary = generate_summary_from_intent(df, intent_config) if intent_config else ""
    local_answer = ""

    # Handle simple fact-based questions locally
    q = question.lower()

    if "how many unique categories" in q:
        local_answer = f"You have {df['Category'].nunique()} unique categories in your inventory."

    elif "list unique categories" in q:
        categories = df['Category'].dropna().unique().tolist()
        local_answer = f"The unique categories in your inventory are: {', '.join(categories)}"

    elif "top categories" in q and "sales" in q:
        top_sales = df.groupby("Category")["My Sales"].sum().sort_values(ascending=False).head(5)
        local_answer = f"Top 5 categories by total sales:\n{top_sales.to_string()}"

    elif matched_row:
        local_answer = f"Here is the row matching the Item ID:\n\n{matched_row}"

    # Show local answer if available
    if local_answer:
        st.subheader("🧮 Local Answer")
        st.markdown(local_answer)

    # GPT fallback
    elif st.button("Analyze with GPT"):
        prompt = f"""
You are a helpful data analyst for my diamonds inventory.

{column_descriptions}

{"Here is a summary based on the full dataset:\n" + dynamic_summary if dynamic_summary else ""}
{"Here is a row matching the Item ID:\n" + matched_row if matched_row else ""}
{"Here is a sample of the dataset:\n" + sample if not matched_row and not dynamic_summary else ""}

User question: {question}

Please answer using the full dataset, not just the sample. Be concise and specific.
"""
        with st.spinner("Analyzing..."):
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a data analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
        answer = response.choices[0].message.content

        # Save to session state
        st.session_state.qa_history.append((question, answer))

        st.subheader("🔍 GPT Analysis")
        st.markdown(response.choices[0].message.content)

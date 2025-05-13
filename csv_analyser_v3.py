import pandas as pd
import streamlit as st
from openai import OpenAI
from difflib import get_close_matches
import re

# 🧠 Set your OpenAI API key (securely with secrets or env var in real apps)
client = OpenAI(
  api_key="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"
)

# 🧑‍💻 Define function for local analysis (for the most common queries)
def get_top_slowest_categories(df):
    return df.groupby("Category")["Aging"].mean().sort_values(ascending=False).head(5)

def get_top_fastest_categories(df):
    return df.groupby("Category")["Aging"].mean().sort_values(ascending=True).head(5)

def get_most_expensive_stone(df):
    return df.loc[df["Total Price"].idxmax()]

def get_cheapest_stone(df):
    return df.loc[df["Total Price"].idxmin()]

def get_sales_by_category(df, sales_column="My Sales"):
    return df.groupby("Category")[sales_column].sum().sort_values(ascending=False)

def get_stock_count(df):
    return df["My Stock"].sum()

def get_item_ids_by_category(df, category):
    return df[df["Category"] == category]["Item Id"].tolist()

# 🔍 Enhanced fuzzy matching utilities
def normalize_category(cat):
    return re.sub(r'[^A-Za-z0-9]+', '', cat).lower()

def extract_item_id(question):
    matches = re.findall(r'\b\d{7,8}\b', question)
    return matches[0] if matches else None

def extract_category(question, category_list):
    norm_cat_map = {normalize_category(cat): cat for cat in category_list}
    norm_cats = list(norm_cat_map.keys())
    question_norm = normalize_category(question)
    close_matches = get_close_matches(question_norm, norm_cats, n=1, cutoff=0.6)
    return norm_cat_map[close_matches[0]] if close_matches else None

def get_price_by_item_id(df, item_id):
    row = df[df["Item Id"] == int(item_id)]
    return row["Total Price"].values[0] if not row.empty else None

# Normalized category matching
def find_best_category_match(user_input, categories):
    user_input = user_input.lower()
    categories_lower = {cat.lower(): cat for cat in categories}
    close_matches = get_close_matches(user_input, categories_lower.keys(), n=1, cutoff=0.6)
    return categories_lower[close_matches[0]] if close_matches else None

# Helper: Extract 7–8 digit numbers (likely item IDs)
def extract_possible_item_id(text):
    match = re.search(r'\b\d{7,8}\b', text)
    return match.group() if match else None


# 📂 Load your CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

column_descriptions = """
Here is a description of key columns in the dataset:

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

# 📝 Keyword-based query map
query_map = {
    "fastest moving categories": get_top_fastest_categories,
    "slowest moving categories": get_top_slowest_categories,
    "most expensive stone": get_most_expensive_stone,
    "cheapest stone": get_cheapest_stone,
    "sold the most": get_sales_by_category,
    "stock count": get_stock_count,
    "item ids": get_item_ids_by_category
}

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df["Item Id"] = df["Item Id"].astype(int)
    df["Total Price"] = df["Total Price"].astype(float)
    st.write("📄 **Sample Data**")
    st.dataframe(df.head(10))

    question = st.text_input("Ask a question about the dataset:")

    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    if st.session_state.qa_history:
        st.subheader("🕓 Chat History")
        for i, (q, a) in enumerate(reversed(st.session_state.qa_history[-10:])):
            st.markdown(f"**Q{i+1}:** **{q}**")
            st.markdown(f"**A{i+1}:** {a}")

    dynamic_summary = None
    if question:
        question_lower = question.lower()

        # 1️⃣ Direct keyword match
        for keyword, func in query_map.items():
            if keyword in question_lower:
                if func == get_item_ids_by_category:
                    # extract category from query
                    categories = df["Category"].dropna().unique()
                    match = find_best_category_match(question, categories)
                    if match:
                        dynamic_summary = func(df, match)
                    else:
                        st.warning("Couldn't find a matching category.")
                    break
                else:
                    dynamic_summary = func(df)
                    break

        # 2️⃣ Fallback: If no match yet, try item ID lookup
        if dynamic_summary is None:
            item_id = extract_possible_item_id(question)
            if item_id and "Item Id" in df.columns and int(item_id) in df["Item Id"].values:
                stone_row = df[df["Item Id"] == int(item_id)].iloc[0]
                price = stone_row.get("Total Price", "N/A")
                dynamic_summary = f"Item ID {item_id} has a total price of {price}."

    if dynamic_summary is None and question:
        prompt = f"""
        You are a helpful data analyst for my diamonds inventory, giving insights to my inventory.

        {column_descriptions}

        Here is a sample of the dataset:

        {df.head(10).to_string()}

        User question: {question}

        Please analyze and answer based on the above.
        """

        if st.button("Analyze with GPT"):
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
                st.session_state.qa_history.append((question, answer))
                st.subheader("🔍 GPT Analysis")
                st.markdown(answer)

    elif dynamic_summary is not None:
        st.subheader("🔍 Local Analysis Results")
        st.write(dynamic_summary)
        st.session_state.qa_history.append((question, str(dynamic_summary)))
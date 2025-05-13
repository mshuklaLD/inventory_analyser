import pandas as pd
import streamlit as st
from openai import OpenAI

# 🧠 Set your OpenAI API key (securely with secrets or env var in real apps)
client = OpenAI(
  api_key="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"
)

# 🧑‍💻 Define function for local analysis (for the most common queries)
def get_top_slowest_categories(df):
    # Sort by 'Aging' (slowest moving = higher aging) and return top 5 categories
    return df.groupby("Category")["Aging"].mean().sort_values(ascending=False).head(5)

def get_top_fastest_categories(df):
    # Sort by 'Aging' (fastest moving = lower aging) and return top 5 categories
    return df.groupby("Category")["Aging"].mean().sort_values(ascending=True).head(5)

def get_most_expensive_stone(df):
    # Find the stone with the highest 'Total Price'
    return df.loc[df["Total Price"].idxmax()]

def get_cheapest_stone(df):
    # Find the stone with the lowest 'Total Price'
    return df.loc[df["Total Price"].idxmin()]

def get_sales_by_category(df, sales_column="My Sales"):
    # Get total sales by category
    return df.groupby("Category")[sales_column].sum().sort_values(ascending=False)

def get_stock_count(df):
    # Get total stock count (sum of 'My Stock' column)
    return df["My Stock"].sum()

def get_item_ids_by_category(df, category):
    # Get Item IDs for a specific category
    return df[df["Category"] == category]["Item Id"].tolist()

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

# 📝 Keyword-based query map (for quick responses)
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
    st.write("📄 **Sample Data**")
    st.dataframe(df.head(10))

    # # Summary
    # summary = df.describe(include='all').to_string()
    # st.write("📊 **Summary Statistics**")
    # st.text(summary)

    # Optional: Custom question
    question = st.text_input("Ask a question about the dataset:")

    # Initialize session state for Q&A history
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    # Display history
    if st.session_state.qa_history:
        st.subheader("🕓 Chat History")
        for i, (q, a) in enumerate(reversed(st.session_state.qa_history[-10:])):  # show last 10
            st.markdown(f"**Q{i+1}:** **{q}**")
            st.markdown(f"**A{i+1}:** {a}")

    # Local analysis based on keyword map
    dynamic_summary = None
    if question:
        # Check for keyword match in the query map
        question_lower = question.lower()
        for keyword, func in query_map.items():
            if keyword in question_lower:
                dynamic_summary = func(df)
                break

    # If no local match, send to GPT for analysis
    if dynamic_summary is None:
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

                # Save to session state
                st.session_state.qa_history.append((question, answer))
                st.subheader("🔍 GPT Analysis")
                st.markdown(answer)
    else:
        # If local analysis, show the result
        st.subheader("🔍 Local Analysis Results")
        st.write(dynamic_summary)
        st.session_state.qa_history.append((question, str(dynamic_summary)))
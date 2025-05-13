import pandas as pd
from intent_analysis import detect_analysis_intent, generate_summary_from_intent

# Load your CSV
df = pd.read_csv("your_data.csv")

# Get user question
user_question = "Which carat range has the most number of slow moving goods?"

# Detect analysis intent
intent_config = detect_analysis_intent(user_question)

# Generate dynamic summary
summary_text = ""
if intent_config:
    summary_text = generate_summary_from_intent(df, intent_config)

# Build OpenAI prompt
prompt = f"""
You are a helpful data analyst.

Here is a sample of the data:

{df.head(5).to_string(index=False)}

Here is a summary based on the question:

{summary_text}

User question: {user_question}

Please analyze and provide insights.
"""

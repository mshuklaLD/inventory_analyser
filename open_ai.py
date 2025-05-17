from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
import os

# 🧠 Set your OpenAI API key (securely with secrets or env var in real apps)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 📂 Load your CSV file
csv_path = "smaller_sample.csv"  # Replace with your actual file path
df = pd.read_csv(csv_path)


sample = df.head(150).to_string()
summary = df.describe(include='all').to_string()

prompt = f"""
You are a helpful data analyst.

Here is a sample of a dataset:

{sample}

And here is a statistical summary of the dataset:

{summary}

Based on this information, provide:
- Key trends
- Anomalies
- Interesting correlations or patterns
- Suggestions for further analysis
"""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a data analyst."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.3,
    max_tokens=800
)

print("\n🔍 GPT Analysis:\n")
print(response.choices[0].message.content)
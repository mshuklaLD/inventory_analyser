# API KEY - 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
from openai import OpenAI
import pandas as pd


client = OpenAI(
  api_key="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"
)

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
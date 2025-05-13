# vector.py

import pandas as pd
import numpy as np
from InstructorEmbedding import INSTRUCTOR
import faiss
import pickle

# 1. Load Data
df = pd.read_csv("smaller_sample.csv")

# 2. Describe each diamond in natural language
def row_to_text(row):
    return (
        f"{row['Shape']} shape, {row['Carat']}ct, Color {row['Color']}, Clarity {row['Clarity']}. "
        f"Priced at ${row['Price Per Ct']}/ct (Total: ${row['Total Price']}). "
        f"Ranked {row['Rank']} among {row['Count']} similar stones in the market. "
        f"Aged {row['Age']} days in inventory. Rap % is {row['Rap %']}."
    )

df['description'] = df.apply(row_to_text, axis=1)

# 3. Initialize the model
model = INSTRUCTOR("hkunlp/instructor-base")
instruction = "Represent the diamond description for semantic search:"

# 4. Create Embeddings
descriptions = df['description'].tolist()
embeddings = model.encode([[instruction, desc] for desc in descriptions])
embeddings = np.array(embeddings).astype("float32")

# 5. Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# 6. Save FAISS index and metadata
faiss.write_index(index, "diamond_index.faiss")

# Save descriptions for later retrieval
with open("diamond_metadata.pkl", "wb") as f:
    pickle.dump({
        "descriptions": descriptions,
        "original_rows": df.to_dict(orient="records")
    }, f)

print("✅ FAISS index and metadata saved.")

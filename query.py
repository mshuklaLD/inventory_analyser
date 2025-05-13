# query.py

import faiss
import pickle
import numpy as np
from InstructorEmbedding import INSTRUCTOR

# Load model and index
model = INSTRUCTOR("hkunlp/instructor-base")
index = faiss.read_index("diamond_index.faiss")

# Load metadata
with open("diamond_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

descriptions = metadata["descriptions"]
original_rows = metadata["original_rows"]

# User query
query = input("Ask a question about your inventory: ")
instruction = "Represent the diamond question for semantic search:"
query_embedding = model.encode([[instruction, query]]).astype("float32")

# Search
D, I = index.search(query_embedding, k=5)

# Show results
print("\nTop results:\n")
for idx in I[0]:
    print(descriptions[idx])
    print(original_rows[idx])
    print("-" * 40)

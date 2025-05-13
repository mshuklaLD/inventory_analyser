import pandas as pd
from InstructorEmbedding import INSTRUCTOR
import chromadb
from chromadb.config import Settings

# Init Chroma client
chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./chroma_db"))

# Load data
df = pd.read_csv("smaller_sample.csv")

# Convert each row into text
def row_to_text(row):
    return (
        f"{row['Shape']} shape, {row['Carat']}ct, Color {row['Color']}, Clarity {row['Clarity']}. "
        f"Priced at ${row['Price Per Ct']}/ct (Total: ${row['Total Price']}). "
        f"Ranked {row['Rank']} among {row['Count']} similar stones in the market. "
        f"Aged {row['Age']} days in inventory. Rap % is {row['Rap %']}."
    )

texts = df.apply(row_to_text, axis=1).tolist()
metadatas = df.to_dict(orient="records")

# Embed using Instructor
model = INSTRUCTOR("hkunlp/instructor-base")
instruction = "Represent the diamond description for semantic search:"
embeddings = model.encode([[instruction, text] for text in texts])

# Create collection
collection = chroma_client.create_collection(name="diamonds", metadata={"hnsw:space": "cosine"})

# Add to Chroma
collection.add(
    documents=texts,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=[f"diamond-{i}" for i in range(len(texts))]
)

chroma_client.persist()
print("✅ ChromaDB index created and saved.")

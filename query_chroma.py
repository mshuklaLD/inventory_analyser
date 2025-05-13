from InstructorEmbedding import INSTRUCTOR
import chromadb
from chromadb.config import Settings

# Load Chroma
chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./chroma_db"))
collection = chroma_client.get_collection(name="diamonds")

# Load model
model = INSTRUCTOR("hkunlp/instructor-base")
instruction = "Represent the diamond question for semantic search:"

# Input query
query = input("Ask a question about your diamonds: ")
embedding = model.encode([[instruction, query]])[0]

# Search
results = collection.query(query_embeddings=[embedding], n_results=5)

# Show results
for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
    print(doc)
    print(meta)
    print("-" * 40)

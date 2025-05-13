from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# Load the CSV file
df = pd.read_csv("smaller_sample.csv")

# Convert each row into a human-readable insight string
def row_to_text(row):
    return (
        f"{row['Shape']} shape, {row['Carat']}ct (in {row['Carat Range']}), Color {row['Color']}, Clarity {row['Clarity']}. "
        f"Rapaport Price: ${row['Rap $/Ct']} per ct, Rapaport Total: ${row['Rap Total']}. "
        f"Asking Price: ${row['$/Ct']} per ct, Total: ${row['Total Price']}. "
        f"Discount from Rapaport: {row['Rap %']}%. "
        f"Inventory Age: {row['Aging']} days. "
        f"Market Rank: {row['Rank']} among {row['Count']} similar stones on RapNet. "
        f"My Inventory Count: {row['My Stock']}, My Sales (Last 2 months): {row['My Sales']}, YTD Sales: {row['My YTD Sales']}."
    )

# Prepare embeddings
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)
# add_documents = True

# Generate LangChain documents if DB doesn't exist
if add_documents:
    documents = []
    ids = []

    for _, row in df.iterrows():
        content = row_to_text(row)
        doc = Document(
            page_content=content,
            metadata={
                "ItemId": row["Item Id"],
                "Shape": row["Shape"],
                "Carat": row["Carat"],
                "Carat Range": row["Carat Range"],
                "Color": row["Color"],
                "Clarity": row["Clarity"],
                "Aging": row["Aging"],
                "Rank": row["Rank"],
                "Count": row["Count"],
                "My Stock": row["My Stock"],
                "My Sales": row["My Sales"],
                "My YTD Sales": row["My YTD Sales"]
            }
        )
        documents.append(doc)
        ids.append(str(row["Item Id"]))

# Create or load ChromaDB
vector_store = Chroma(
    collection_name="inventory_analysis",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

# Expose retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 250})
# retriever = vector_store.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={'score_threshold': 0.8}
# )

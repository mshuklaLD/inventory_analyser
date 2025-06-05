from elasticsearch import Elasticsearch, helpers
from elasticsearch.helpers import bulk

import pandas as pd

# Load and normalize data
df = pd.read_csv("smaller_sample.csv")
# Replace NaN values with "None" for specific fields (fluor, shade, milky)
df['fluor'] = df['Fluor'].apply(lambda x: "None" if pd.isna(x) else x)
df['shade'] = df['Shade'].apply(lambda x: "None" if pd.isna(x) else x)
df['milky'] = df['Milky'].apply(lambda x: "None" if pd.isna(x) else x)

# Normalize field names safely
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("/", "_per_")
    .str.replace("%", "pct")
)

records = df.to_dict(orient="records")

# Connect to Elasticsearch
es = Elasticsearch("http://34.41.92.221:9200")

# Delete index if it exists
if es.indices.exists(index="inventory_nl"):
    es.indices.delete(index="inventory_nl")

# Create new index with mappings
es.indices.create(index="inventory_nl", body={
    "mappings": {
        "properties": {
            "item_id": {"type": "long"},
            "cert_number": {"type": "keyword"},
            "stock_num": {"type": "keyword"},
            "shape": {"type": "keyword"},
            "carat": {"type": "float"},
            "carat_range": {"type": "keyword"},
            "color": {"type": "keyword"},
            "clarity": {"type": "keyword"},
            "cut": {"type": "keyword"},
            "total_price": {"type": "double"},
            "my_stock": {"type": "integer"},
            "my_sales": {"type": "integer"},
            "my_ytd_sales": {"type": "integer"},
            "rap_pct": {"type": "float"},
            "rank": {"type": "integer"},
            "category": {"type": "keyword"},
            "lab": {"type": "keyword"},
            "fluor": {"type": "keyword"},
            "polish": {"type": "keyword"},
            "sym": {"type": "keyword"},
            "shade": {"type": "keyword"},
            "milky": {"type": "keyword"}
        }
    }
})
print(records[0])

# Bulk ingest data
actions = [
    {"_index": "inventory_nl", "_source": record}
    for record in records
]

success, failed = bulk(es, actions, raise_on_error=False)
if failed:
    for item in failed:
        print(f"Failed to index document: {item}")
else:
    print(f"✅ Successfully ingested {len(actions)} records.")

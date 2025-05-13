from elasticsearch import Elasticsearch, helpers
import json

es = Elasticsearch("http://localhost:9200")

# Delete old index if it exists
if es.indices.exists(index="inventory"):
    es.indices.delete(index="inventory")

# Create new index with mappings
es.indices.create(index="inventory", body={
    "mappings": {
        "properties": {
            "status": { "type": "keyword" },
            "price": { "type": "double" },
            "sold_at": { "type": "date" },
            "shape": { "type": "keyword" },
            "carat": { "type": "float" }
        }
    }
})

with open("sample_data.json") as f:
    data = json.load(f)

actions = [
    { "_index": "inventory", "_source": item }
    for item in data
]

helpers.bulk(es, actions)
print("✅ Sample data ingested.")

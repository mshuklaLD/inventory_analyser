from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")

print("Connected:", es.ping())
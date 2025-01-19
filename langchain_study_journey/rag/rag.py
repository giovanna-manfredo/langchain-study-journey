from qdrant_client import QdrantClient
import config
qdrant_client = QdrantClient(
    url="",
    api_key="",
)

print(qdrant_client.get_collections())
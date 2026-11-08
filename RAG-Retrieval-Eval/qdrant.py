import config
from qdrant_client import QdrantClient

CLIENT = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_KEY)
COLLECTION = "session2_realdocs"

import os
from dotenv import load_dotenv

load_dotenv(".env", override=True)

os.environ.setdefault("API_KEY", os.getenv("API_KEY", ""))
os.environ.setdefault(
    "LANGFUSE_BASE_URL", os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
)

LLM_MODEL = os.getenv("LLM_MODEL", "gemini/gemini-flash-latest")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_KEY = os.getenv("QDRANT_API_KEY")
DOCS_DIR = "session2_docs"

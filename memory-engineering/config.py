import os
from dotenv import load_dotenv
load_dotenv(".env", override=True)

os.environ.setdefault("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
LLM_MODEL = os.getenv("LLM_MODEL", "gemini/gemini-flash-latest")
USER_ID = "applied-ai-demo-user"
SUPERMEMORY_API_KEY = os.getenv("SUPERMEMORY_API_KEY","")
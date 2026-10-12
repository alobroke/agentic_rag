
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
QDRANT_PATH = BASE_DIR / "data" / "qdrant"

COLLECTION_NAME = "agentic_knowledge"

# Local embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# open router hosted LLM
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "nvidia/nemotron-3.5-lightning:free"
)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Optional web search
WEB_SEARCH_API_KEY = os.getenv("TAVILY_API_KEY", "")

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
SQLITE_DB_PATH = DATA_DIR / "papers.db"
CHROMA_DB_PATH = DATA_DIR / "chroma_db"

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Anthropic
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")

# arXiv
ARXIV_API_BASE = "http://export.arxiv.org/api/query"
ARXIV_RATE_LIMIT_SECONDS = 3.0
ARXIV_MAX_RESULTS = 100

# Ranking
RANKING_BATCH_SIZE = 15
RANKING_TOP_K = 5

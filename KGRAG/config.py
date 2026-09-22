import os
from pathlib import Path
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

# ============================================================
# API KEYS & CREDENTIALS
# ============================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Neo4j Aura Database Credentials
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# ============================================================
# DIRECTORY & FILE PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

RAW_DATA_PATH = BASE_DIR / "schemes.json"
CLEAN_DATA_PATH = BASE_DIR / "schemes_cleaned.json"
CLEAN_CSV_PATH = BASE_DIR / "schemes_cleaned.csv"
FAISS_INDEX_DIR = BASE_DIR / "faiss_index"

# ============================================================
# SCRAPER SETTINGS
# ============================================================
BASE_URL = "https://www.tn.gov.in/"
DEPARTMENT_URL = "https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="

# ============================================================
# MODEL CONFIGURATIONS
# ============================================================
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o"
LLM_TEMPERATURE = 0.0
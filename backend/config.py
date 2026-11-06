import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env file
load_dotenv(BASE_DIR / ".env")

# API Keys (loaded securely from .env or system environment)
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Gemini Model
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# Database path (stored in local workspace or data folder)
DB_NAME = os.getenv("DB_PATH", "indian_stock_market.sqlite")
DB_PATH = BASE_DIR / DB_NAME

# Eligible security document file names
FILE_EQUITY_L = BASE_DIR / "EQUITY_L.csv"
FILE_SME_EQUITY_L = BASE_DIR / "SME_EQUITY_L.csv"
FILE_ELIGIBLE_T0 = BASE_DIR / "Eligible_T0_Securities_20260930121437.xlsx"

# App Defaults
DEFAULT_START_DATE = os.getenv("DEFAULT_START_DATE", "2015-01-01")
DEFAULT_MAX_STOCKS = int(os.getenv("DEFAULT_MAX_STOCKS", "50"))
DEFAULT_PAUSE_SECONDS = float(os.getenv("DEFAULT_PAUSE_SECONDS", "0.4"))
DEFAULT_SKIP_EXISTING = True


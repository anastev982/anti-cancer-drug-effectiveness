from pathlib import Path

# Define project root: src/config.py → ../../
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_PROCESSED_DIR = DATA_DIR / "raw_data" / "processed"
PROCESSED_DIR = DATA_DIR / "processed"
FINAL_CHUNKS_DIR = PROJECT_ROOT / "final_chunks"

# Parquet output path
PARQUET_OUTPUT_PATH = PROCESSED_DIR / "final_merged.parquet"

# (Optional) for compatibility
RAW_DATA_DIR = RAW_PROCESSED_DIR
NEW_PROCESSED_DATA_DIR = PROJECT_ROOT / "processed"
NEW_PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

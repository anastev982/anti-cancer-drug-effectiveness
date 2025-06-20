import pandas as pd
from pathlib import Path

# Path to the data folder
data_dir = Path("data/raw_data/processed")

# List of all the files in the folder
csv_files = list(data_dir.glob("*.csv"))

for file_path in csv_files:
    print(f"\nFile: {file_path.name}")
    try:
        df = pd.read_csv(file_path, nrows=5)  # loads just the first 5 rows
        print(f"Columns ({len(df.columns)}):\n{df.columns.tolist()}")
        print("Sample data:")
        print(df.head(), "\n")
    except Exception as e:
        print(f"Cannot load file: {e}")

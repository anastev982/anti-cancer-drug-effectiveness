import pandas as pd
from pathlib import Path


RAW_PATH = Path("data/raw_data/processed")
CLEANED_PATH = Path("cleaned")
CLEANED_PATH.mkdir(parents=True, exist_ok=True)


def clean_row_data_one():
    """Load and clean row_data_one.csv and save as Parquet."""
    csv_path = RAW_PATH / "row_data_one.csv"
    df = pd.read_csv(csv_path)

    print(" Original shape:", df.shape)
    print(" Columns:", df.columns.tolist())

    # 1. Strip leading/trailing spaces from column names
    df.columns = df.columns.str.strip()

    # 2. Drop any fully empty rows
    df.dropna(how="all", inplace=True)

    # 3. Drop duplicates if any
    df.drop_duplicates(inplace=True)

    # 4. Optional: convert Drug ID to numeric
    df["Drug ID"] = pd.to_numeric(df["Drug ID"], errors="coerce")

    print("Cleaned shape:", df.shape)

    # Save cleaned file
    out_path = CLEANED_PATH / "cleaned_row_data_one.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved cleaned data to: {out_path}")


if __name__ == "__main__":
    clean_row_data_one()

import pandas as pd
from pathlib import Path
import sys
from pathlib import Path

# Add the root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))


def clean_drug_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\n Cleaning Drug Data...")

    # Drop duplicate rows (if any)
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"- Dropped {before - after} duplicate rows")

    # Drop columns with more than 90% missing values
    threshold = 0.9 * len(df)
    before_cols = df.shape[1]
    df = df.dropna(thresh=threshold, axis=1)
    after_cols = df.shape[1]
    print(f"- Dropped {before_cols - after_cols} mostly-empty columns")

    # Fill remaining NaNs with sentinel or mean value depending on dtype
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna("unknown")

    print(f"- Final shape: {df.shape}")
    return df


def clean_gene_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\n Cleaning Gene Data...")

    # Drop duplicate rows
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"- Dropped {before - after} duplicate rows")

    # Drop columns with more than 90% missing values
    threshold = 0.9 * len(df)
    before_cols = df.shape[1]
    df = df.dropna(thresh=threshold, axis=1)
    after_cols = df.shape[1]
    print(f"- Dropped {before_cols - after_cols} mostly-empty columns")

    # Fill remaining NaNs
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna("unknown")

    print(f"- Final shape: {df.shape}")
    return df


if __name__ == "__main__":
    from src.load_bouth_datasets import drug_df, gene_df

    drug_df_cleaned = clean_drug_data(drug_df)
    gene_df_cleaned = clean_gene_data(gene_df)

    # Optional: Save cleaned versions
    Path("cleaned").mkdir(exist_ok=True)
    drug_df_cleaned.to_parquet("cleaned/cleaned_drug_data.parquet")
    gene_df_cleaned.to_parquet("cleaned/cleaned_gene_data.parquet")
    print("\n Cleaned data saved to ./cleaned/")

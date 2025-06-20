#!/usr/bin/env python3
"""
Download FPKM data files from the GDC API based on a manifest.

- Retries up to 3 times for network reliability.
- Unzips .gz files automatically and removes the compressed archive.
"""

from __future__ import annotations

import gzip
import os
import shutil
import time
from pathlib import Path

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_fixed
from tqdm import tqdm


# Configuration


MANIFEST_PATH = Path("data/tcga_brca_fpkm_manifest.csv")
OUTPUT_DIR = Path("data/fpkm_files")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Download logic


@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def download_and_unzip(file_id: str, filename: str) -> None:
    """
    Download a file from GDC and unzip it if necessary.

    Retries 3 times (with 5s pause) on network errors.
    """
    url = f"https://api.gdc.cancer.gov/data/{file_id}"
    local_path = OUTPUT_DIR / filename
    unzipped_path = local_path.with_suffix("")

    if unzipped_path.exists():
        return  # Already downloaded & unzipped

    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(response.content)

        # If it’s a .gz file, unzip it
        if local_path.suffix == ".gz":
            with gzip.open(local_path, "rb") as f_in:
                with open(unzipped_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            local_path.unlink()  # Remove .gz file after extraction

    except requests.exceptions.RequestException as e:
        print(f"Request failed for {filename}: {e}")
        raise
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        raise


# Main pipeline


def main() -> None:
    # Load manifest
    df = pd.read_csv(MANIFEST_PATH)
    print(f"Loaded manifest with {len(df)} entries.")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading"):
        file_id = row["id"]
        filename = row["filename"]
        download_and_unzip(file_id, filename)

    print(f"All downloads complete. Files saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

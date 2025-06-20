#!/usr/bin/env python3
"""Entry point for the GDSC large-scale analysis pipeline."""

from __future__ import annotations
import argparse, logging, sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import dask
from dask.distributed import Client, LocalCluster
import dask.dataframe as dd

from src.helpers import check_required_columns
from src.dataloader import load_csv
from src.helpers import RAW_DATA_DIR
from src.merge_utils import merged_drug_data, merged_gene_data

dask.config.set({"dataframe.shuffle.method": "disk"})
dask.config.set({"temporary-directory": "./dask-temp"})

DEFAULT_LOG_FILE = Path("app.log")
MERGED_DRUG_PATH = Path("merged_drug_data.parquet")
MERGED_GENE_PATH = Path("merged_gene_data.parquet")


def setup_logging(log_file: Path = DEFAULT_LOG_FILE) -> logging.Logger:
    logger = logging.getLogger("gdsc_pipeline")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.addHandler(logging.StreamHandler(stream=sys.stdout))
    logger.addHandler(logging.FileHandler(log_file, mode="a"))
    logging.getLogger("distributed").setLevel(logging.WARNING)
    return logger


def get_dask_client(
    *, dashboard: bool = True, n_workers: int = 1, threads_per_worker: int = 1
) -> Client:
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        diagnostics_port=8787 if dashboard else None,
        silence_logs=logging.ERROR,
    )
    return Client(cluster)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full GDSC analysis pipeline.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--force-remerge", action="store_true")
    return parser.parse_args(argv)


def save_separately(drug_df, gene_df):
    if not drug_df.compute().empty:
        drug_df.to_parquet(MERGED_DRUG_PATH)
        print("Saved merged_drug_data.parquet")
    else:
        print("drug_df is empty — skipping save.")

    if not gene_df.compute().empty:
        gene_df.to_parquet(MERGED_GENE_PATH)
        print("Saved merged_gene_data.parquet")
    else:
        print("gene_df is empty — skipping save.")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logger = setup_logging()

    logger.info("=== Starting GDSC cleaning and separate merge pipeline ===")

    with get_dask_client(
        dashboard=not args.no_dashboard,
        n_workers=args.workers,
        threads_per_worker=args.threads,
    ) as client:
        logger.info("Dask client ready. Dashboard -> %s", client.dashboard_link)

        # Skip merging if files already exist
        if (
            MERGED_DRUG_PATH.exists()
            and MERGED_GENE_PATH.exists()
            and not args.force_remerge
        ):
            print(" Merged files already exist. Skipping merge.")
            logger.info("Merged files already exist. Skipping merge.")
            drug_df = dd.read_parquet(MERGED_DRUG_PATH)
            gene_df = dd.read_parquet(MERGED_GENE_PATH)
        else:
            logger.info("Loading raw data files…")
            drug_list = load_csv(RAW_DATA_DIR / "drug_list.csv")
            gdsc1 = load_csv(RAW_DATA_DIR / "gdsc1.csv")
            gdsc2 = load_csv(RAW_DATA_DIR / "gdsc2.csv")
            variance = load_csv(RAW_DATA_DIR / "variance.csv")
            mutations = load_csv(RAW_DATA_DIR / "mutations.csv")
            all_mutations = load_csv(RAW_DATA_DIR / "all_mutations.csv")

            check_required_columns(all_mutations, ["model_id"], "all_mutations")
            check_required_columns(gdsc1, ["drug_id", "cell_line_name"], "gdsc1")
            check_required_columns(gdsc2, ["drug_id", "cell_line_name"], "gdsc2")
            check_required_columns(drug_list, ["drug_id"], "drug_list")

            logger.info("Merging drug data…")
            drug_df = merged_drug_data(drug_list, gdsc1, gdsc2)

            logger.info("Merging gene data…")
            gene_df = merged_gene_data(variance, mutations, all_mutations)

            logger.info("Saving merged data separately…")
            save_separately(drug_df, gene_df)

        # Summary
        logger.info("=== Pipeline finished successfully ===")
        print("\n DRUG DATA:")
        print(f"Rows: ~{len(drug_df)} | Columns: {len(drug_df.columns)}")
        print(drug_df.dtypes)
        print(drug_df.head())

        print("\n GENE DATA:")
        print(f"Rows: ~{len(gene_df)} | Columns: {len(gene_df.columns)}")
        print(gene_df.dtypes)
        print(gene_df.head())


if __name__ == "__main__":
    main()

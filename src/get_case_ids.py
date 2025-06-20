#!/usr/bin/env python3
"""
Fetch FPKM-level expression file metadata from GDC for TCGA-BRCA.

Saves a CSV manifest with file_id, filename, and submitter_id.
"""

from __future__ import annotations

import csv
import requests


def main() -> None:
    url = "https://api.gdc.cancer.gov/files"
    params = {
        "filters": {
            "op": "and",
            "content": [
                {
                    "op": "in",
                    "content": {
                        "field": "cases.project.project_id",
                        "value": ["TCGA-BRCA"],
                    },
                },
                {
                    "op": "in",
                    "content": {
                        "field": "data_category",
                        "value": ["Transcriptome Profiling"],
                    },
                },
                {
                    "op": "in",
                    "content": {
                        "field": "data_type",
                        "value": ["Gene Expression Quantification"],
                    },
                },
                {
                    "op": "in",
                    "content": {
                        "field": "analysis.workflow_type",
                        "value": ["HTSeq - FPKM"],
                    },
                },
            ],
        },
        "fields": "file_id,file_name,cases.submitter_id",
        "format": "JSON",
        "size": "2000",
    }

    print(" Querying GDC API for TCGA-BRCA FPKM metadata…")
    response = requests.post(url, json=params)
    response.raise_for_status()  # Will raise HTTPError if the request failed

    data = response.json()

    results = []
    for file in data.get("data", {}).get("hits", []):
        results.append(
            {
                "id": file["file_id"],
                "filename": file["file_name"],
                "submitter_id": file["cases"][0]["submitter_id"],
            }
        )

    out_file = "data/tcga_brca_fpkm_manifest.csv"
    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "filename", "submitter_id"])
        writer.writeheader()
        writer.writerows(results)

    print(f" Saved FPKM manifest to {out_file} ({len(results)} records)")


if __name__ == "__main__":
    main()

import requests
import csv

# Step 1: Get file IDs for FPKM expression files from TCGA-BRCA
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
    "size": 10000,  # Big enough to get them all
}

response = requests.post(url, json=params)
# Check if the 'data' and 'hits' keys exist in the response before processing
if response.status_code == 200:
    data = response.json()

    # Save file IDs and names to manifest.csv
    if "data" in data and "hits" in data["data"]:
        results = []
        for file in data["data"]["hits"]:
            results.append(
                {
                    "id": file["file_id"],
                    "filename": file["file_name"],
                    "submitter_id": file["cases"][0]["submitter_id"],
                }
            )

        with open("tcga_brca_fpkm_manifest.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "filename", "submitter_id"])
            writer.writeheader()
            writer.writerows(results)

        print("Saved FPKM manifest to data/tcga_brca_fpkm_manifest.csv")
    else:
        print("API response did not contain 'data' or 'hits'.")
else:
    print(f"Failed to fetch data from GDC API. Status code: {response.status_code}")

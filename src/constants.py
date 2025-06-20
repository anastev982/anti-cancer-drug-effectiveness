# src/constants.py

# Columns expected in the drug list dataframe
DRUG_COLUMNS = [
    "drug_id",
    "name",
    "targets",
    "target_pathway",
    "synonyms",
    "datasets",
    "pubchem",
]

# Columns expected in GDSC drug screening dataframes
GDSC_COLUMNS = [
    "drug_id",
    "cell_line_name",
    "conc",
    "intensity",
    "date_created",
    "drugset_id",
    "duration",
]

# Default gene columns for scaling
DEFAULT_GENE_COLS = [
    "tp53",
    "brca1",
    "egfr",
]

# Gene-related columns
GENE_ID = "gene_id"
SYMBOL = "symbol"
VARIANT_SYMBOL = "variant_symbol"

# Possible categorical bins for concentration column
CONC_BINS = ["Low", "Medium", "High"]

# Columns often useful for mutation data merges
MUTATION_COLUMNS = [
    GENE_ID,
    "mutation_id",
    "mutation_effect",
    "sample_id",
]

# For feature engineering: numeric columns that might need scaling or transformation
NUMERIC_FEATURES = [
    "intensity",
    "conc",
]

# Columns related to drug response metrics, useful for aggregations
DRUG_RESPONSE_COLUMNS = [
    "drug_id",
    "intensity",
    "mean_intensity",
]

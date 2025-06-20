import pyarrow.parquet as pq

# Load just the Parquet schema
print("Loading schema...")
schema = pq.read_schema("final_features/full_model_input.parquet")
print(f"Number of columns: {len(schema.names)}")
print(schema.names[:20])  # show only first 20

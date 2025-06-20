import pandas as pd

df = pd.read_csv("output/clean_data.csv")

print("Basic info")
print(df.info())

print("\n First Few Rows:")
print(df.head())

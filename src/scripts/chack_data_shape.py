def check_nulls(df):
    print("Missing values per column:")
    print(df.isnull().sum())


def check_shape(df):
    print(f"DataFrame shape: {df.shape}")

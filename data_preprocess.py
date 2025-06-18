import polars as pl

def load_data(path: str, file_path:str) -> pl.DataFrame:
    df = pl.read_parquet(file_path)
    df = preprocess_data(df)

def preprocess_data(df: pl.DataFrame) -> pl.DataFrame:
    df = df.drop_nulls()
    return df


from phm_ml.config.config_loader import DataConfig
DATA_CONFIG = DataConfig.from_yaml("data.yaml")
print(DATA_CONFIG)

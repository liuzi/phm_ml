import polars as pl
from phm_ml.data_process.data_backblaze_harddrive import data_clean
from phm_ml.config.config_loader import DataConfig

scan_df = pl.scan_parquet(DataConfig.path['train_data'])
df = data_clean(scan_df)
print(df.slice(0,50).collect())
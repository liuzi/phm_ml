import polars as pl
from phm_ml.data_process.data_backblaze_harddrive import data_clean, get_disk_serials, adjust_dates, fix_date_gaps, create_failed_sequences, create_normal_sequences 
from phm_ml.config.config_loader import DataConfig
from phm_ml.utils.logging import setup_logging  

logger = setup_logging()

train_df = pl.scan_parquet(DataConfig.path['train_data'])
normal_serials, failed_serials, normal_df, failed_df = get_disk_serials(train_df, num_normal=100, logger=logger)

normal_df = fix_date_gaps(data_clean(normal_df, logger), normal_serials, logger)
failed_df = fix_date_gaps(data_clean(failed_df, logger), failed_serials, logger)
failed_sequences = create_failed_sequences(failed_df, failed_serials, sequence_length=5, lookahead=1, logger=logger)
normal_sequences = create_normal_sequences(normal_df, normal_serials, sequence_length=5, lookahead=1, logger=logger)

print(failed_sequences.slice(0,50).collect())
print(normal_sequences.slice(0,50).collect())

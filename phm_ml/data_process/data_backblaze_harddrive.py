# This script is based on "https://github.com/HROlive/Applications-of-AI-for-Predictive-Maintenance/blob/main/Lab2-LSTM-For-Timeseries.ipynb"
import logging
import polars as pl
from typing import Tuple

def data_clean(df: pl.LazyFrame, logger: logging.Logger) -> pl.LazyFrame:
    """
    Clean the input dataframe by removing unnecessary columns and preprocessing SMART sensor data.
    
    Args:
        df: Input LazyFrame containing hard drive data
        
    Returns:
        Cleaned LazyFrame
    """
    logger.info("Removing unnecessary columns")
    
    # Collect schema once to avoid repeated schema resolution
    schema = df.collect_schema()
    column_names = schema.names()
    
    # Identify columns to drop - only include columns that actually exist
    potential_drop_columns = ['model', 'capacity_bytes'] # drop model since we've already filtered specific model ST4000DM000
    normalized_columns = [col for col in column_names if col.endswith("normalized")]
    drop_columns = [col for col in potential_drop_columns + normalized_columns if col in column_names]
    
    # Identify SMART columns that actually exist
    smart_columns = [col for col in column_names if col.startswith("smart_") and col not in drop_columns]
    
    # Remove unnecessary columns only if they exist
    if drop_columns:
        logger.info(f"Dropping columns: {drop_columns}")
        df = df.drop(drop_columns)
    else:
        logger.info("No columns to drop")

    logger.info("Converting smart sensor columns to float64")
    # Convert all smart sensor columns to float64 only if they exist
    if smart_columns:
        logger.info(f"Converting {len(smart_columns)} SMART columns to float64")
        df = df.with_columns(pl.col(smart_columns).cast(pl.Float64))
    else:
        logger.warning("No SMART columns found")
    
    logger.info("Sorting by timestamp and serial_number(SN)")
    # Sort by timestamp and serial_number(SN) and fill nulls
    df = df.sort(["date", "serial_number"]).fill_null(0)

    return df


def get_disk_serials(df: pl.LazyFrame, num_normal_serials: int = 100, logger: logging.Logger = None) -> Tuple[pl.Series, pl.Series]:
    """
    Extract serial numbers of failed and normal disks.
    
    Args:
        df: Input LazyFrame containing hard drive data
        num_normal_serials: Number of normal disks to select (default: 100)
        
    Returns:
        Tuple of (failed_serials, normal_serials) as Polars Series
    """
    # Get the disk serials of the failed disks
    failed_df = df.filter(pl.col("failure") == 1)
    failed_serials = failed_df.select("serial_number").unique().collect().to_series()
    
    if logger:
        logger.info(f"Number of failed disks: {failed_serials.len()}")

    # Get normal disk serials (excluding failed ones)
    normal_df = df.filter(~pl.col('serial_number').is_in(failed_serials))
    normal_serials = (normal_df
                     .group_by('serial_number')
                     .agg(pl.count().alias('count'))
                     .sort('count', descending=True)
                     .head(num_normal_serials)
                     .select('serial_number')
                     .collect()
                     .to_series())
    
    if logger:
        logger.info(f"Number of normal disks: {normal_serials.len()}")
    return normal_serials, failed_serials, normal_df, failed_df


def adjust_dates(df: pl.LazyFrame, logger: logging.Logger = None) -> pl.LazyFrame:
    """
    Adjust dates in the input dataframe to fill missing dates.
    
    Args:
        df: Input LazyFrame containing hard drive data
        logger: Optional logger instance for logging
        
    Returns:
        LazyFrame with adjusted dates
    """
    if logger:
        logger.info("Adjusting dates to fill missing dates")
    
    # Get basic info (need to collect minimal data for logic)
    cur_serial = df.select(pl.col('serial_number')).unique().collect().to_series().to_list()[0]
    col_list = df.collect_schema().names()  # Fix performance warning by using collect_schema()
    
    # Get date range info
    date_stats = df.select([
        pl.col('date').min().alias('first_date'),
        pl.col('date').max().alias('last_date'),
        pl.col('date').count().alias('record_count')
    ]).collect()
    
    first_date = date_stats['first_date'][0]
    last_date = date_stats['last_date'][0]
    record_count = date_stats['record_count'][0]
    
    # Calculate expected number of days
    num_date_range = (last_date - first_date).days + 1
    
    # Check if we need to fill missing dates
    if num_date_range > record_count:
        # Collect the data for processing (needed for complex date filling logic)
        df_collected = df.sort('date').collect()
        
        # Create complete date range and ensure it matches the input date type
        date_range = pl.date_range(
            first_date, 
            last_date, 
            interval='1d',
            eager=True
        ).to_frame('date').with_columns([
            pl.col('date').cast(df_collected.schema['date'])  # Cast to match input date type
        ])
        
        # Join with original data to identify missing dates
        df_complete = (
            date_range
            .join(
                df_collected, 
                on='date', 
                how='left'
            )
            # Add the serial number to missing rows
            .with_columns(
                pl.when(pl.col('serial_number').is_null())
                .then(pl.lit(cur_serial))
                .otherwise(pl.col('serial_number'))
                .alias('serial_number')
            )
            # Forward fill all columns except date and serial_number
            .with_columns([
                pl.col(col).forward_fill() 
                for col in col_list 
                if col not in ['date', 'serial_number']
            ])
        )
        
        return df_complete.lazy()
    else:
        # No missing dates, return original LazyFrame
        return df


def fix_date_gaps(df: pl.LazyFrame, serials: pl.Series, logger: logging.Logger = None) -> pl.LazyFrame:
    """
    Fix date gaps in the input dataframe for all serial numbers.
    
    Args:
        df: Input LazyFrame containing hard drive data
        serials: Series of disk serial numbers
        logger: Optional logger instance for logging
        
    Returns:
        LazyFrame with fixed date gaps
    """
    if logger: 
        logger.info("Fixing date gaps for all serial numbers")
    
    # Process each serial number and collect results
    fixed_frames = []
    
    for cur_serial in serials.to_list():
        # Filter for current serial and process
        serial_lf = df.filter(pl.col('serial_number') == cur_serial)
        fixed_serial_lf = adjust_dates(serial_lf)
        fixed_frames.append(fixed_serial_lf)
    
    # Concatenate all processed LazyFrames
    if fixed_frames:
        df_fixed = pl.concat(fixed_frames, how='vertical')
        return df_fixed
    else:
        # Return empty LazyFrame with same schema if no serials provided
        return df.filter(pl.lit(False))
    

def create_failed_sequences(df: pl.LazyFrame, failed_serials: pl.Series, sequence_length: int, lookahead: int, logger: logging.Logger = None) -> pl.LazyFrame:
    """
    Create sequences of data from the dataframe for failed disks.
    
    Args:
        df: Input LazyFrame containing hard drive data
        failed_serials: Series of failed disk serial numbers
        sequence_length: Length of each sequence
        lookahead: Number of steps to look ahead
        logger: Optional logger instance for logging
        
    Returns:
        LazyFrame with sequences of data
    """

    logger.info(f"Creating sequences of data for {failed_serials.len()} failed disks")

    failed_seq_list = []
    for serial in failed_serials:
        tmp_serial_failed = df.filter(pl.col('serial_number') == serial).collect()

        if tmp_serial_failed.shape[0] > (sequence_length + lookahead):
            tmp_serial_failed = tmp_serial_failed.with_columns(
                pl.arange(pl.len()).over('serial_number').alias('row_idx'))
            
            failed_rows = tmp_serial_failed.filter(pl.col('failure') == 1)
            end_idx = failed_rows.select('row_idx').row(0)[0] - lookahead +1
            start_idx = end_idx - sequence_length

            if start_idx > 0:
                failed_seq_list.append(tmp_serial_failed.slice(start_idx, sequence_length).lazy())
    del tmp_serial_failed
    
    if logger:
        logger.info(f"Created {len(failed_seq_list)} sequences of data for {failed_serials.len()} failed disks")
    
    return pl.concat(failed_seq_list, how='vertical') # TODO: delete row index


def create_normal_sequences(df: pl.LazyFrame, normal_serials: pl.Series, sequence_length: int, lookahead: int, logger: logging.Logger = None) -> pl.LazyFrame:
    """
    Create sequences of data from the dataframe for normal disks.
    
    Args:
        df: Input LazyFrame containing hard drive data
        normal_serials: Series of normal disk serial numbers        
        sequence_length: Length of each sequence
        lookahead: Number of steps to look ahead
        logger: Optional logger instance for logging
        
    Returns:
        LazyFrame with sequences of data
    """ 

    logger.info(f"Creating sequences of data for {normal_serials.len()} normal disks")

    for serial in normal_serials:
        df_tmp = df.filter(pl.col('serial_number') == serial).collect()
        if df_tmp.shape[0] > (sequence_length + lookahead):
            df_tmp = df_tmp.with_columns(
                pl.arange(pl.len()).over('serial_number').alias('row_idx'))
            
            normal_rows = df_tmp.filter(pl.col('failure') == 0)
            end_idx = normal_rows.select('row_idx').row(0)[0] - lookahead +1        
            ## TODO: correct





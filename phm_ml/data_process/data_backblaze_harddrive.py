import logging
import polars as pl

def data_clean(df:pl.DataFrame):
    logging.info("Removing unecessary columns")
    # remove normalized columns
    drop_columns = [col for col in df.columns if col.endswith("normalized")]+['model', 'capacity_bytes']
    df = df.drop(drop_columns)

    logging.info("Converting smart sensor columns to float64")
    # convert all smart sensor columns to float64
    df = df.with_columns(pl.col([col for col in df.columns if col.startswith("smart_")])
                         .cast(pl.Float64))
    
    logging.info("Sorting by timestamp and serial_number(SN)")
    # sort by timestamp and serial_number(SN)
    df = df.sort(["date", "serial_number"])
    df = df.fill_null(0)

    return df

def create_sequences(df:pl.DataFrame, sequence_length:int=10):
    """
    Create sequences of data from the dataframe.
    """
    sequences = []
    for i in range(len(df) - sequence_length + 1):
        sequences.append(df.iloc[i:i+sequence_length])





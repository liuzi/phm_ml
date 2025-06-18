## TODO: Merge this script (rawdata -> first version of processed data)
import polars as pl
import glob
import re
from pathlib import Path

def load_all_csv_files_streaming(base_directory='.', year=2017, num_quarters=2, num_csv=None):
    """
    Load and combine CSV files from specified directories using Polars for memory efficiency
    
    Args:
        base_directory: Base directory containing the data folders
        year: Year of data to load
        num_quarters: Number of quarters to load
        num_csv: Optional limit on number of CSV files to load
    """
    base_path = Path(base_directory)
    matching_directories = list(base_path.glob(f'data_*_{str(year)}'))

    if not matching_directories:
        print(f"No directories matching 'data_*_{str(year)}' found in {base_directory}")
        return None

    def directory_sort_key(dir_path):
        dir_name = dir_path.name
        match = re.search(r'data_(.+)_{str(year)}', dir_name)
        if match:
            middle_part = match.group(1)
            q_match = re.match(r'Q(\d+)', middle_part)
            if q_match:
                return int(q_match.group(1))
            return middle_part
        return dir_name

    sorted_directories = sorted(matching_directories, key=directory_sort_key)

    print(f"Found {len(sorted_directories)} matching directories (sorted):")
    for directory in sorted_directories:
        print(f"  - {directory}")

    combined_df = None
    csv_count = 0
    for directory in sorted_directories[:num_quarters]:
        csv_files = sorted(directory.glob("*.csv"))
        print(f"\nLoading {len(csv_files)} CSV files from {directory}:")

        for file in csv_files:
            try:
                df = pl.read_csv(file)
                if df.height == 0:
                    print(f"  - Skipped empty or all-NaN file: {file}")
                    continue

                csv_count += 1
                print(f"  - Loaded {file.name}: ({df.height} rows, {df.width} columns)")

                if combined_df is None:
                    combined_df = df
                else:
                    combined_df = pl.concat([combined_df, df], how="vertical_relaxed", rechunk=True)
                    
                if num_csv and csv_count == num_csv:
                    return combined_df
                del df

            except Exception as e:
                print(f"  - Error loading {file}: {str(e)}")

    print(f"\nSuccessfully loaded {csv_count} CSV files from {num_quarters} directories")
    return combined_df

def preprocess_data(df, harddrive_model='ST4000DM000'):
    """
    Preprocess the data for a specific hard drive model
    
    Args:
        df: Input Polars DataFrame
        harddrive_model: Model to filter for
    """
    print("Initial preprocessing steps...")
    
    # Filter for specific model
    df = df.filter(pl.col("model") == harddrive_model)
    
    # Convert date to datetime
    df = df.with_columns(pl.col("date").str.to_datetime())
    
    # Drop unnecessary columns
    df = df.drop(["model", "date", "serial_number"])
    
    # Remove normalized columns (keep only raw values)
    cols = [c for c in df.columns if c.lower().find("normalized") == -1]
    df = df[cols]
    
    # Remove zero-valued columns
    df = df.select([col for col in df.columns if df[col].is_not_null().any()])
    
    # Fill any remaining nulls with 0
    df = df.fill_null(0)
    
    return df

def balance_dataset(df, balance_ratio=1.2):
    """
    Balance the dataset by downsampling the majority class
    
    Args:
        df: Input Polars DataFrame
        balance_ratio: Ratio of normal to failed samples
    """
    print("Balancing dataset...")
    
    # Get failed samples
    failed_samples = df.filter(pl.col("failure") == 1)
    n_failed = failed_samples.height
    
    # Calculate number of normal samples to keep
    n_normal_target = int(n_failed * balance_ratio)
    
    # Get random sample of normal cases
    normal_samples = df.filter(pl.col("failure") == 0).sample(n=n_normal_target)
    
    # Combine failed and sampled normal cases
    balanced_df = pl.concat([failed_samples, normal_samples])
    
    print(f"Original dataset size: {df.height}")
    print(f"Balanced dataset size: {balanced_df.height}")
    print(f"Number of failed samples: {n_failed}")
    print(f"Number of normal samples: {n_normal_target}")
    
    return balanced_df

def prepare_train_test_split(df):
    """
    Prepare features and target variables
    
    Args:
        df: Input Polars DataFrame
    """
    # Separate features and target
    X = df.drop("failure")
    y = df.select("failure")
    
    return X, y

def main():
    # Example usage
    base_dir = "/path/to/data"
    
    # Load training data
    print("Loading training data...")
    df_train = load_all_csv_files_streaming(base_dir, year=2017, num_quarters=1)
    
    # Load test data
    print("\nLoading test data...")
    df_test = load_all_csv_files_streaming(base_dir, year=2016, num_quarters=1)
    
    # Preprocess both datasets
    df_train = preprocess_data(df_train)
    df_test = preprocess_data(df_test)
    
    # Balance datasets
    df_train = balance_dataset(df_train)
    df_test = balance_dataset(df_test)
    
    # Prepare features and targets
    X_train, y_train = prepare_train_test_split(df_train)
    X_test, y_test = prepare_train_test_split(df_test)
    
    print("\nPreprocessing complete!")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    main() 
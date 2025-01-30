import os
import pandas as pd

def combine_parquet_files(directory):
    """
    Combine all parquet files in a given directory into a single pandas DataFrame.

    Parameters:
    directory (str): The directory path where the parquet files are located.

    Returns:
    pandas.DataFrame: A DataFrame that combines all the parquet files in the directory.
    """
    dfs = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.parquet'):
                filepath = os.path.join(root, file)
                df = pd.read_parquet(filepath)
                dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def read_raw_data():
    """
    Read raw data from parquet files in a specific directory, combine them, 
    and write the combined data back to a parquet file.
    """
    directory = 'data/short_summary'
    combined_df = combine_parquet_files(directory)
    combined_df.to_parquet('data/output/combined.parquet', index=False)

if __name__ == "__main__":
    read_raw_data()

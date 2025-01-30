# src/split_dataset.py

import pandas as pd
import argparse

def split_dataset(file_path, cutoff_date, save=False, output_dir='data/output/'):
    """
    Split the dataset based on a cutoff date for calculate_date.
    Parameters:
        file_path (str): Path to the input dataset file.
        cutoff_date (str): The cutoff date to split the training and testing sets.
        save (bool): Whether to save the split datasets to disk.
        output_dir (str): Directory where the split datasets will be saved.
    """
    # Load the dataset
    df = pd.read_parquet(file_path)
    # Specify target and predictors
    target = 'target_host_sold-today'
    predictors = df.drop(columns=[target]).columns

    # Convert cutoff_date string to datetime
    cutoff_date = pd.Timestamp(cutoff_date)
    # Split the data based on 'calculate_date'
    train_df = df[df['calculate_date'] < cutoff_date]
    test_df = df[df['calculate_date'] >= cutoff_date]

    # Separate predictors and target for training and testing sets
    X_train = train_df[predictors]
    y_train = train_df[target].to_frame()
    X_test = test_df[predictors]
    y_test = test_df[target].to_frame()

    if save:
        # Save the split datasets to parquet files
        X_train.to_parquet(f'{output_dir}X_train.parquet')
        X_test.to_parquet(f'{output_dir}X_test.parquet')
        y_train.to_parquet(f'{output_dir}y_train.parquet')
        y_test.to_parquet(f'{output_dir}y_test.parquet')

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset with a specified cutoff date")
    parser.add_argument('--cutoff_date', type=str, required=True, help="Cutoff date in YYYY-MM-DD format")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = split_dataset(
        file_path='data/output/feature_engineered.parquet',
        cutoff_date=args.cutoff_date,
        save=True
    )

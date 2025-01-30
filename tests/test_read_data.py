import os
import pandas as pd
import pytest
from src.read_data import combine_parquet_files, read_raw_data

def test_combine_parquet_files(tmpdir):
    # Create temporary parquet files for testing
    df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    df2 = pd.DataFrame({'A': [7, 8, 9], 'B': [10, 11, 12]})
    df1.to_parquet(os.path.join(tmpdir, 'file1.parquet'))
    df2.to_parquet(os.path.join(tmpdir, 'file2.parquet'))

    # Use the function to combine these files
    result_df = combine_parquet_files(tmpdir)

    # Check if the resulting DataFrame is correct
    expected_df = pd.concat([df1, df2], ignore_index=True)
    result_df = result_df.sort_values(by='A').reset_index(drop=True)
    expected_df = expected_df.sort_values(by='A').reset_index(drop=True)
    pd.testing.assert_frame_equal(result_df, expected_df)

def test_read_raw_data_functionality(tmpdir, monkeypatch):
    # Create a temporary parquet file
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    df.to_parquet(os.path.join(tmpdir, 'file1.parquet'))

    # Mocking the directory reading to only consider the tmpdir
    def mock_combine(*args, **kwargs):
        return pd.read_parquet(os.path.join(tmpdir, 'file1.parquet'))
    
    monkeypatch.setattr('src.read_data.combine_parquet_files', mock_combine)

    # Running the function which should use the mocked directory and file
    read_raw_data()

    # Read the output file and check if it is correct
    output_df = pd.read_parquet('data/output/combined.parquet')
    pd.testing.assert_frame_equal(output_df, df)

# Pytest fixture to automatically manage temporary directory for the test duration
@pytest.fixture
def temp_dir(tmpdir):
    original_dir = os.getcwd()
    os.chdir(tmpdir)
    yield tmpdir
    os.chdir(original_dir)


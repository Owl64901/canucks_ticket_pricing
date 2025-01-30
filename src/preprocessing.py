# src/preprocessing.py

from datetime import datetime
import pandas as pd
import argparse
import re

# Global variables
PRESEASON_RANGES = [
    ('2022-09-24', '2022-10-08'),
    ('2023-09-23', '2023-10-07')
]
EXCLUDED_EVENTS = ['Canucks vs. TBD', 'Test Event - DO NOT PURCHASE']
DAYS_BEFORE_EVENT = 30

def extract_and_format_date(df):
    """
    Extracts the event date from the identifier and formats it in 'YYYY-MM-DD' format.
    """
    df['event_date'] = df['identifier'].str.extract(r'((?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)_(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)_[0-9]{1,2}_\d{4})_')
    df['event_date'] = pd.to_datetime(df['event_date'], format='%a_%b_%d_%Y')
    df['event_date'] = df['event_date'].dt.strftime('%Y-%m-%d')
    return df

def extract_opponent_from_event(df):
    """
    Extracts the opponent team name from the event name.
    """
    df['opponent'] = df['event_name'].map(extract_opponent)
    return df

def extract_opponent(event_name):
    """
    Helper function to extract the opponent team name from a given event name.
    """
    if (event_name.startswith("Canucks")):
        match = re.search(r'vs\.\s(.+)', event_name)
        if (match):
            return match.group(1)
    return event_name

def extract_price_code(df):
    """
    Extracts the price code from the price level name.
    """
    df['price_code'] = df['price_level_name'].str[-1]
    return df

def filter_date_name(df, start_date, excluded_events=EXCLUDED_EVENTS):
    """
    Filters the DataFrame to include only records after a specified start date
    and excludes certain event names.
    """
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df[df['event_date'] > pd.to_datetime(start_date)]
    df = df[~df['event_name'].isin(excluded_events)]
    return df

def drop_invalid_host_sold(df, threshold=0):
    """
    Drops rows where 'host_sold-yesterday' is less than a specified threshold.
    """
    df = df[df['host_sold-yesterday'] >= threshold]
    return df

def shift_target(df):
    """
    Shifts the 'host_sold-yesterday' column by -1 to create the 'target_host_sold-today' column.
    """
    df = df.sort_values('calculate_date')
    df['target_host_sold-today'] = df.groupby(['event_id', 'price_code'])['host_sold-yesterday'].shift(-1)
    df = df[df['target_host_sold-today'] >= 0]
    df = df.dropna(subset=['target_host_sold-today'])
    return df

def add_previous_days_sold(df, days_to_add=7):
    """
    Adds 7 new columns to the DataFrame for tickets sold 1 to 7 days before today.
    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    Returns:
    df (pandas.DataFrame): The DataFrame with added columns for tickets sold in previous days.
    """
    df = df.sort_values(['event_id', 'calculate_date'])
    for i in range(2, days_to_add + 1):
        df[f'tickets_sold_{i}_days_before_today'] = df.groupby(['event_id', 'price_code'])['target_host_sold-today'].shift(i)
    return df

def filter_within_30_days(df, days_before_event=DAYS_BEFORE_EVENT):
    """
    Filters the DataFrame to include only records where 'calculate_date' is within a specified number of days prior to 'event_date'.
    """
    df['event_date'] = pd.to_datetime(df['event_date'])
    df['calculate_date'] = pd.to_datetime(df['calculate_date'])
    df = df[df['event_date'] - df['calculate_date'] <= pd.Timedelta(days=days_before_event)]
    return df

def drop_price_code_v(df):
    """
    Drops data with price_code = 'V'.
    """
    df = df[df['price_code'] != 'V']
    return df

def fill_initial_price(df):
    """
    Fills 'last_price' with 'initial_price' where 'last_price' is 0.
    """
    df.loc[df['last_price'] == 0, 'last_price'] = df['initial_price']
    return df

def filter_out_preseason(df, preseason_ranges=PRESEASON_RANGES):
    """
    Filters out rows where 'event_date' falls within the specified preseason date ranges.
    """
    df['event_date'] = pd.to_datetime(df['event_date'])
    for start_date, end_date in preseason_ranges:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        df = df[~((df['event_date'] >= start_date) & (df['event_date'] <= end_date))]
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data with a specified full capacity date")
    parser.add_argument('--full_capacity_date', type=str, required=True, help="Full capacity date in YYYY-MM-DD format")
    args = parser.parse_args()

    df = pd.read_parquet('data/output/combined.parquet')
    df['calculate_date'] = pd.to_datetime(df['calculate_date'], format='%m/%d/%Y')
    df = extract_and_format_date(df)
    df = extract_price_code(df)
    df = drop_price_code_v(df)
    df = filter_date_name(df, args.full_capacity_date)
    df = filter_out_preseason(df)
    df = extract_opponent_from_event(df)
    df = drop_invalid_host_sold(df)
    df = shift_target(df)
    df = add_previous_days_sold(df)
    df = filter_within_30_days(df)
    df = fill_initial_price(df)
    df['ticket_sold-total'] = df['ticket_sold-total'].astype('int64')
    columns_to_drop = ['venue', 'platinum_opens', 'platinum_average_open_price', 'platinum_lowest_open_price', 'platinum_highest_open_price', 'platinum_holds',
                       'dl_rowhash', 'dl_iscurrent', 'dl_isdeleted', 'dl_recordstartdateutc',
                       'dl_recordenddateutc', 'dl_load_id', 'archtics_event_name', 'host_event_code']
    df = df.drop(columns=columns_to_drop)
    df.to_parquet('data/output/processed.parquet')

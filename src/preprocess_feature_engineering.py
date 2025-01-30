# src/preprocess_feature_engineering.py

import pandas as pd
import warnings
import sys
warnings.filterwarnings('ignore')

from src.preprocessing import (
    extract_and_format_date,
    extract_opponent_from_event,
    extract_price_code,
    filter_date_name,
    drop_invalid_host_sold,
    add_previous_days_sold,
    filter_within_30_days,
    drop_price_code_v,
    filter_out_preseason
)

from src.feature_engineering import (
    add_weekend_month_column,
    add_days_until_game_column,
    add_opponent_popularity,
    add_vancouver_ranking,
    add_bowl_location,
    add_host_sold_agg,
    add_ga_data,
    add_price_code_ordinal,
    add_price_floor_and_times_above_floor,
    add_inventory_normalized
)

def shift_target(df):
    df = df.sort_values('calculate_date')
    df['target_host_sold-today'] = df.groupby(['event_id', 'price_code'])['host_sold-yesterday'].shift(-1)
    # Custom behavior: No filtering or dropping NaNs
    return df

def fill_initial_price(df):
    df.loc[df['last_price'] == 0, 'last_price'] = df.loc[df['last_price'] == 0, 'initial_price']
    return df

def preprocess_data(df, start_date):
    df['calculate_date'] = pd.to_datetime(df['calculate_date'])
    df = extract_and_format_date(df)
    df = extract_price_code(df)
    df = drop_price_code_v(df)
    df = filter_date_name(df, start_date)
    df = filter_out_preseason(df)
    df = extract_opponent_from_event(df)
    df = drop_invalid_host_sold(df)
    df = shift_target(df)
    df = add_previous_days_sold(df)
    df = filter_within_30_days(df)
    df = fill_initial_price(df)
    return df

def feature_engineer_data(df):
    df = add_weekend_month_column(df)
    df = add_days_until_game_column(df)
    df = add_opponent_popularity(df)
    df = add_vancouver_ranking(df)
    df = add_bowl_location(df)
    df = add_host_sold_agg(df)
    df = add_ga_data(df)
    df = add_price_code_ordinal(df) 
    df = add_price_floor_and_times_above_floor(df)
    df = add_inventory_normalized(df)
    return df

def main(input_file, output_file, start_date):
    df = pd.read_parquet(input_file)
    df = preprocess_data(df, start_date)
    df = feature_engineer_data(df)
    df.to_parquet(output_file)

if __name__ == "__main__":
    input_file = 'data/output/combined.parquet'
    output_file = 'data/output/feature_engineered.parquet'
    start_date = '2022-02-15'  # Default start date if not provided
    main(input_file, output_file, start_date)

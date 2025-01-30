import pandas as pd

def add_weekend_month_column(df):
    """
    This function adds a binary column 'weekend_game' and a 'month' column to the DataFrame df.
    The column 'weekend_game' indicates whether the game is on a weekend (Friday, Saturday, or Sunday).
    The column 'month' indicates the month of the event_date.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame.

    Returns:
    df (pandas.DataFrame): The DataFrame with the added 'weekend_game' and 'month' columns.
    """
    df['event_date'] = pd.to_datetime(df['event_date']) 
    df['weekend_game'] = df['event_date'].dt.dayofweek.isin([4, 5, 6]).astype(int)
    df['month'] = df['event_date'].dt.month
    return df


def add_days_until_game_column(df):
    """
    This function adds a new column 'days_until_game' to the DataFrame df.
    The column 'days_until_game' represents the number of days from 'calculate_date' to 'event_date'.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame.

    Returns:
    df (pandas.DataFrame): The DataFrame with the added 'days_until_game' column.
    """
    df['calculate_date'] = pd.to_datetime(df['calculate_date'])
    df['event_date'] = pd.to_datetime(df['event_date'])
    df['days_until_game'] = (df['event_date'] - df['calculate_date']).dt.days
    return df

def add_opponent_popularity(df):
    """
    This function appends a new column 'opponent_rank' to the dataframe.
    The column 'opponent_rank' represents the popularity of the opponent team, either 1, 2, or 3. 
    This function is dependent on the existing 'opponent' column (obtained from preprocessing.extract_opponent)
    and the opponent_popularity.csv data. 
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame. 
    Returns: 
    df (pandas.DataFrame): The DataFrame with the added 'opponent_rank' column.
    """

    df_ranking = pd.read_csv('data/opponent_popularity.csv')
    merged_df = pd.merge(df, df_ranking[['data_name', 'opponent_rank']], left_on='opponent', right_on='data_name', how='left')
    merged_df['opponent_rank'] = merged_df['opponent_rank'].fillna(-1).astype(int)
    return merged_df

def add_vancouver_ranking(df):
    """
    This function appends a new column 'van_rank' to the dataframe.
    The column 'van_rank' represents the canucks updated standings at the most recent monday.    
    This function is dependent on the vancouver_ranking.csv data. 
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame. 
    Returns: 
    df (pandas.DataFrame): The DataFrame with the added 'opponent_ranking' column.
    """

    df = df.sort_values(by='calculate_date') 
    df_rank = pd.read_csv('data/vancouver_ranking.csv') 
    df_rank['rank_date'] = pd.to_datetime(df_rank['rank_date'])
    
    # merge dataframes, fill NAs and clean
    df = pd.merge(df, df_rank, left_on='calculate_date', right_on='rank_date', how='left')
    df['van_rank'] = df['van_rank'].ffill().bfill()
    df['van_rank'] =  df['van_rank'].astype(int)
    df.drop('rank_date', axis=1,inplace=True)
    return df 

def add_bowl_location(df):
    """
    This function adds a new binary column 'bowl_location' to the DataFrame df.
    The column 'bowl_location' indicates whether the seating area is in the lower bowl (1) or upper bowl (0).

    Parameters:
    df (pandas.DataFrame): The input DataFrame.

    Returns:
    df (pandas.DataFrame): The DataFrame with the added 'bowl_location' column.
    """
    lower_bowl_codes = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C'}
    df['bowl_location'] = df['price_code'].apply(lambda x: 1 if x in lower_bowl_codes else 0)
    return df

def add_host_sold_agg(df):
    """
    This function adds two new columns 'host_sold_agg_last_day' and 'host_sold_agg_last_day_samebowl' to the DataFrame df.
    The column 'host_sold_agg_last_day' is the sum of 'host_sold-yesterday' for each group of 'event_id' and 'calculate_date'.
    The column 'host_sold_agg_last_day_samebowl' is the sum of 'host_sold-yesterday' for each group of 'event_id', 'calculate_date'
    and 'bowl_location'. 

    Parameters:
    df (pandas.DataFrame): The input DataFrame.

    Returns:
    df (pandas.DataFrame): The DataFrame with the columns 'host_sold_agg_last_day' and 'host_sold_agg_last_day_samebowl' added.
    """
    df['host_sold_agg_last_day'] = df.groupby(['event_id', 'calculate_date'])['host_sold-yesterday'].transform('sum')
    df['host_sold_agg_last_day_samebowl'] = df.groupby(['event_id', 'calculate_date', 'bowl_location'])['host_sold-yesterday'].transform('sum')
    return df

def add_ga_data(df): 
    """
    This function adds 3 columns 'unique_views', 'order_qty', 'ticket_qty' from google analytics (ga_data.csv) to the DataFrame df.
    Note: the values in each column represent the overall value for the calculate_date (not broken into price_level).

    Parameters:
    df (pandas.DataFrame): The input DataFrame.

    Returns:
    df (pandas.DataFrame): The DataFrame with the 3 added columns: 'unique_views', 'order_qty' and 'ticket_qty'. 
    """  
    # read and clean ga data 
    ga_df = pd.read_csv('data/ga_data.csv')
    ga_df = ga_df[ga_df['Event_Name'].str.startswith('Canucks')].copy()
    ga_df['as_of_date'] = pd.to_datetime(ga_df['as_of_date'], format='%Y-%m-%d')
    ga_df['Event_Date'] = pd.to_datetime(ga_df['Event_Date'], format='%Y-%m-%d')
    ga_df = ga_df[ga_df['as_of_date'] <= ga_df['Event_Date']].copy()

    # aggregate duplicate rows 
    agg_dict = {
        'Event_Name': 'first',
        'Event_Client_Id': 'first',
        'Venue_Name': 'first',
        'Event_Time': 'first',
        'unique_views': 'max',
        'order_qty': 'max',
        'tkt_qty': 'max'
    }
    df_agg = ga_df.groupby(['Event_Date', 'as_of_date'], as_index=False).agg(agg_dict)

    # merge data 
    df = pd.merge(df, df_agg , left_on=['event_date', 'calculate_date'], right_on=['Event_Date', 'as_of_date'], how='left')
    df = df.drop(columns = ['Event_Date', 'as_of_date', 'Event_Client_Id', 'Venue_Name', 'Event_Time', 'Event_Name'])
    return df  

def add_price_code_ordinal(df):
    """
    This function adds a new column 'price_code_ordinal' to the DataFrame df.
    The column 'price_code_ordinal' is an ordinal representation of the 'price_code' column.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.

    Returns:
    df (pandas.DataFrame): The DataFrame with the added 'price_code_ordinal' column.
    """
    price_code_dict = {
        '0': 31, 'S': 30, '3': 29, '1': 28, '2': 27, 'U': 26, '4': 25, '5': 24, '6': 23, 'T': 23,
        '7': 21, 'D': 20, '8': 19, 'H': 18, '9': 17, 'O': 16, 'A': 15, 'E': 14, 'B': 13, 'F': 13,
        'I': 11, 'J': 10, 'C': 9, 'P': 8, 'G': 7, 'K': 6, 'M': 5, 'L': 4, 'Q': 3, 'N': 2, 'R': 1
    }
    df['price_code_ordinal'] = df['price_code'].map(price_code_dict)
    return df
    
def add_price_floor_and_times_above_floor(df):
    """
    This function adds two new columns 'price_floor' and 'times_above_floor' to the DataFrame df.
    The column 'price_floor' is the floor price for each price code.
    The column 'times_above_floor' is the ratio of 'last_price' to 'price_floor'.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.

    Returns:
    df (pandas.DataFrame): The DataFrame with the added 'price_floor' and 'times_above_floor' columns.
    """
    price_floor_dict = {
        '0': 185.00, 'S': 159.00, '3': 149.00, '1': 147.00, '2': 138.00,
        'U': 132.00, '4': 123.00, '5': 115.00, '6': 100.00, 'T': 100.00,
        '7': 93.00, 'D': 92.00, '8': 86.00, 'H': 83.00, '9': 82.00, 'O': 79.00,
        'A': 76.00, 'E': 74.00, 'B': 68.00, 'F': 68.00, 'I': 65.00,
        'J': 65.00, 'C': 58.00, 'P': 55.00, 'G': 54.00, 'K': 51.00,
        'M': 47.00, 'L': 45.00, 'Q': 44.00, 'N': 38.00, 'R': 37.00
    }
    df['price_floor'] = df['price_code'].map(price_floor_dict)
    df['times_above_floor'] = df['last_price'] / df['price_floor']
    return df

def add_inventory_normalized(df):
    """
    This function adds a new column 'inventory_normalized' to the DataFrame df.
    This column represents the inventory ('pminventory') normalized by the capacity ('cap') per price level.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame.

    Returns:
    df (pandas.DataFrame): The DataFrame with the added 'host_sold_agg' column.
    """
    df['inventory_normalized'] = df['pminventory']/df['cap'] 

    return df 

if __name__ == "__main__":
    """
    This is the main function that reads in the processed.parquet file,
    applies the functions within this script, and saves the resulting DataFrame
    to a new parquet file named feature_engineered.parquet.
    """
    df = pd.read_parquet('data/output/processed.parquet')
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
    df.to_parquet('data/output/feature_engineered.parquet')

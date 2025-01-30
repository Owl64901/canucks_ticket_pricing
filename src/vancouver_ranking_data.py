
"""
Module Name: vancouver_ranking_data.py
Date: 2024-06-21

This script pulls the league standings data from the NHL API at given dates and 
generates a csv file with the Canucks ranking. It is dependent on 
data/output/processed.parquet. 
"""
import pandas as pd
import requests
import os 

def main():
    """
    Main function to orchestrate the process of fetching NHL standings data,
    determining Canucks ranking, and exporting results to CSV.
    """
    # Setup and load required data 
    rank_file_path = 'data/vancouver_ranking.csv'
    processed_file_path = 'data/output/processed.parquet'
    pro_df = pd.read_parquet(processed_file_path)
    pro_df = pro_df.sort_values(by='calculate_date')

    # Read existing ranking data, if it exists
    if os.path.isfile(rank_file_path):
        existing_rank_df = pd.read_csv(rank_file_path)
    else: 
        existing_rank_df = pd.DataFrame(columns=['rank_date', 'van_rank'])

    # Retrieve dates to fetch new data for 
    dates = get_dates(pro_df, existing_rank_df)

    # Fetch the data 
    if len(dates) != 0 :
        van_rank_df = vancouver_ranking(dates)
    else:
        van_rank_df = pd.DataFrame()

    # Append new data and export to csv  
    if not van_rank_df.empty:
        append_to_csv(existing_rank_df, van_rank_df, rank_file_path) 
    else:
        print("No new data to write to csv.")


def get_dates(pro_df, existing_rank_df):
    """
    Extracts unique Monday dates from pro_df and ensures the start date is included
    if it's not a Monday. Removes dates already present in existing_rank_df.

    Parameters
    ----------
    pro_df : pandas.DataFrame
        DataFrame containing processed data with 'calculate_date' column.
    existing_rank_df : pandas.DataFrame
        DataFrame containing existing rank data with 'rank_date' column.

    Returns
    -------
    list
        List of dates to fetch new data from NHL API.
    """
    # Obtain monday dates from df 
    calc_dates = pro_df['calculate_date'].unique()  
    monday_dates = [date.strftime('%Y-%m-%d') for date in calc_dates if date.weekday() == 0]

    # Insert processed data start date, if its not a monday 
    start_date = pro_df['calculate_date'].iloc[0]
    if start_date.weekday() != 0:
        monday_dates.insert(0, start_date.strftime('%Y-%m-%d'))

    # Remove dates already in existing_df 
    existing_dates = set(existing_rank_df['rank_date'])
    fetch_dates = [date for date in monday_dates if date not in existing_dates]
    print(f'Number of dates to fetch: {len(fetch_dates)} \nDates to fetch: {fetch_dates}')
  
    return fetch_dates

def vancouver_ranking(dates):
    """
    Fetches standings data for Canucks from NHL API for given dates.
    Returns a DataFrame with Canucks ranking data, containing 
    two columns: 'rank_date' and 'van_rank'. 

    Parameters
    ----------
    dates : list
        List of dates in '%Y-%m-%d' format to retrieve data for.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing Canucks ranking data with columns 'rank_date' and 'van_rank'.
    """
    van_rank = pd.DataFrame(columns=['rank_date', 'van_rank']) # to store retrieved rankings  
    failed_dates = []

    for date in dates:
        
        api_endpoint = f'https://api-web.nhle.com/v1/standings/{date}'
        response = requests.get(api_endpoint)
       
        # Check if request is successful 
        status_messages = {
            400: "Bad Request - The server could not understand the request due to invalid syntax.",
            401: "Unauthorized - Authentication is required and has failed.",
            403: "Forbidden - The client does not have access rights to the content.",
            404: "Not Found - The server cannot find the requested resource.",
            429: "Rate Limit Exceeded - Too many requests in the time duration.",
            500: "Internal Server Error - The server encountered an unexpected condition.",
            503: "Service Unavailable - The server is not ready to handle the request.",
            }

        if response.status_code != 200:
            print(f'Request was unsuccessful - unable to fetch data for: {date}')
            print(f'Request error code: {response.status_code}')
            if response.status_code in status_messages: 
                print(f'Error: {status_messages[response.status_code]}')
                failed_dates.append(date)
            continue

        data = response.json() 

        # Check if data is empty and skip date if so  
        if data['standings'] == []:
            print(f'No data available for: {date}')
            failed_dates.append(date)   # record failed date
            continue 

        # Convert required data to a pandas dataframe
        team_standings = []
        for team_data in data['standings']:
            team_standings.append({
                'team_name': team_data['teamName']['default'],
                'points': team_data['points']
            })
        team_standings_df = pd.DataFrame(team_standings)

        # Determine Canucks ranking
        team_standings_df['van_rank'] = team_standings_df['points'].rank(method='min', ascending=False).astype(int)

        # Keep date and Canucks ranking only 
        van_df = team_standings_df[team_standings_df['team_name'] == 'Vancouver Canucks'].copy()
        van_df.loc[:, 'rank_date'] = date
        van_df = van_df[['rank_date', 'van_rank']]
        
        # Append Canucks ranking to final dataframe 
        van_rank = pd.concat([van_rank, van_df], ignore_index=True)
        
        print(f'Data Success for: {date}')

    # Print request summary  
    if failed_dates != []:
        print(f'Request Complete. \nUnable to fetch data for {len(failed_dates)} out of {len(dates)} dates.\nFailed dates: \n{failed_dates}')
    else: 
        print(f'Request Complete.')

    return van_rank  

    
def append_to_csv(existing_rank_df, van_rank_df, rank_file_path):
    """
    Appends van_rank_df to existing_rank_df, sorts by rank_date, and exports to CSV.

    Parameters
    ----------
    existing_rank_df : pandas.DataFrame
        DataFrame containing existing Canucks ranking data. 
    van_rank_df : pandas.DataFrame
        DataFrame containing new Canucks ranking data to append.
    rank_file_path : str
        File path to which the updated DataFrame is exported as a CSV file.
    """  
    updated_df = pd.concat([existing_rank_df, van_rank_df], ignore_index=True)
    updated_df = updated_df.sort_values(by='rank_date')
    updated_df.to_csv(rank_file_path, index=False)   

    print(f'Data exported to {rank_file_path}')

if __name__ == "__main__":
    main()
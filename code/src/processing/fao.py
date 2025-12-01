from src.processing.functions import filter_data
from src.config import COUNTRIES, YEARS

def process_temp_change(df):
    filtered_df = filter_data(df, 'area', COUNTRIES)
    filtered_df = filter_data(filtered_df, 'year', YEARS)

    # process std deviation and temperature change here
    pivoted_df = filtered_df.pivot_table(
            index=['area', 'year'],          # keep one row per country-year
            columns='element',               # pivot the 'measure' column
            values='value'                   # fill values from 'value' column
        ).reset_index()
        
    # optional: rename columns for convenience
    pivoted_df = pivoted_df.rename(columns={
        'Temperature change': 'temp_change',
        'Standard Deviation': 'std_dev_temp_change',
        'area': 'country'
    })
    
    return pivoted_df

def process_forest(df):
    return df

def process_production(df):
    return df


from itertools import count
import pandas as pd
from src.processing.functions import filter_data
from src.config import COUNTRIES, YEARS, GOOGLE_DRIVE, COUNTRIES_ALT

def process_temp_change(df):
    # rename countries based on COUNTRIES_ALT
    df['area'] = df['area'].replace(COUNTRIES_ALT)
    
    filtered_df = filter_data(df, 'area', COUNTRIES)
    filtered_df = filter_data(filtered_df, 'year', YEARS)

    assert len(filtered_df['area'].unique()) == len(COUNTRIES), "Not all countries are present after filtering."

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
    filtered_df = filter_data(df, 'area', COUNTRIES)
    filtered_df = filter_data(filtered_df, 'year', YEARS)

    # load hs2-cpc mapping
    processed_dir = GOOGLE_DRIVE / "processed"
    hs6cpc = pd.read_csv(processed_dir / "hs6cpc_mapping.csv")

    print(hs6cpc.head())
    print(filtered_df.head())

    # Remove apostrophes, ensure strings
    filtered_df['item code (cpc)'] = (
        filtered_df['item code (cpc)']
        .astype(str)
        .str.replace("'", "", regex=False)
        .str.strip()
    )

    hs6cpc['cpc product code'] = (
        hs6cpc['cpc product code']
        .astype(str)
        .str.strip()
    )

    hs6cpc['cpc product code'] = hs6cpc['cpc product code'].astype(str)
    hs6cpc['hs 2002 product code'] = hs6cpc['hs 2002 product code'].astype(str)

    merged_df = filtered_df.merge(
        hs6cpc[['hs 2002 product code', 'cpc product code']],
        left_on='item code (cpc)',
        right_on='cpc product code',
        how='left'
    )

    merged_df = merged_df.rename(columns={"hs 2002 product code": "hs6"})

    # print rows where hs6 is null
    print("Rows with null hs6 after merge:")
    print(merged_df[merged_df['hs6'].isnull()])
    # count
    print(f"Number of rows with null hs6: {merged_df['hs6'].isnull().sum()}")

    # remove rows where hs6 is null
    merged_df = merged_df[merged_df['hs6'].notnull()]

    print(merged_df.head())

    return merged_df


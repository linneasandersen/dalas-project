import pandas as pd
from src.fetch.countries import fetch_countries
from src.processing.functions import filter_data
from src.config import COUNTRIES, YEARS, GOOGLE_DRIVE, COUNTRIES_ALT

def process_rta(df):

    countries_df = fetch_countries()

    # replace country names with their alternative names if needed
    countries_df['country'] = countries_df['country'].replace(COUNTRIES_ALT)

    COUNTRY_CODES = countries_df['alpha3'].where(countries_df['country'].isin(COUNTRIES)).tolist()
    # remove nans from country codes
    COUNTRY_CODES = [code for code in COUNTRY_CODES if pd.notna(code)]

    assert len(COUNTRY_CODES) == len(COUNTRIES), "Mismatch in number of country codes"
    
    filtered_df = filter_data(df, 'exporter', COUNTRY_CODES)
    filtered_df = filter_data(filtered_df, 'year', YEARS)
    filtered_df = filter_data(filtered_df, 'importer', COUNTRY_CODES)
    # optional: rename columns for convenience
    filtered_df = filtered_df.rename(columns={
        'exporter': 'exporter_code',
        'importer': 'importer_code',
        'year': 'year',
    })

    # replace country codes with country names
    code_to_name = countries_df.set_index('alpha3')['country'].to_dict()
    filtered_df['exporter_name'] = filtered_df['exporter_code'].map(code_to_name)
    filtered_df['importer_name'] = filtered_df['importer_code'].map(code_to_name)

    return filtered_df
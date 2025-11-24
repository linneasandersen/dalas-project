from src.processing.functions import filter_data
from src.config import COUNTRIES

def process_oec(df):
    filtered_df = filter_data(df, 'importer country', COUNTRIES)
    filtered_df = filter_data(filtered_df, 'exporter country', COUNTRIES)

    return filtered_df


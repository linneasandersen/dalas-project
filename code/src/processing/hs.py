from src.processing.functions import filter_data

def process_hs(df):
    filtered_df = filter_data(df, 'product level', 'HS2')

    return filtered_df

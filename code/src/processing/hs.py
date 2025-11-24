from src.processing.functions import filter_data
from src.config import HS2_CODES

def process_hs(df):
    filtered_df = filter_data(df, 'product level', ['HS2'])
    print(HS2_CODES)
    filtered_df = filter_data(filtered_df, 'hs id', [str(code) for code in HS2_CODES])

    return filtered_df

def update_config(df):
    # get the 'oec id' from df
    oec_ids = df['oec id'].unique().tolist()

    return oec_ids
    

from src.processing.functions import filter_data
from src.config import COUNTRIES, HS2_CODES

def process_oec(df):
    filtered_df = filter_data(df, 'importer_name', COUNTRIES)
    filtered_df = filter_data(filtered_df, 'exporter_name', COUNTRIES)

    # filter to only include HS2 codes in HS_CODES
    # Convert HS2 list to 2-digit strings
    hs2_list = [f"{code:02d}" for code in HS2_CODES]

    # Extract HS2
    filtered_df['hs2'] = filtered_df['hs_code'].astype(str).str[:2]

    # Filter to selected HS2 codes
    filtered_df = filtered_df[filtered_df['hs2'].isin(hs2_list)]

    # Aggregate to HS2 level per year/exporter/importer
    filtered_df = (
        filtered_df
        .groupby([
            'year',
            'exporter_id',
            'exporter_name',
            'importer_id',
            'importer_name',
            'hs2'
        ])
        .agg({
            'value': 'sum',
            'quantity': 'sum'
        })
        .reset_index()
    )



    return filtered_df


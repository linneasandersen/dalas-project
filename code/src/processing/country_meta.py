from src.processing.functions import filter_data
from src.config import COUNTRIES, HS2_CODES, YEARS, COUNTRIES_ALT

def process_gdp(df):
    # change country names to match COUNTRIES list, use COUNTRIES_ALT dict
    df['country name'] = df['country name'].replace(COUNTRIES_ALT)

    filtered_df = filter_data(df, 'country name', COUNTRIES)

    # drop unnamed columns
    filtered_df = filtered_df.drop(columns=[col for col in filtered_df.columns if 'unnamed' in col])

    # reshape from wide to long
    long_df = filtered_df.melt(
        id_vars=['country name', 'country code', 'indicator name', 'indicator code'],  # columns to keep as-is
        var_name='year',           # new column for year
        value_name='gdp'           # new column for values
    )

    long_df['year'] = long_df['year'].astype(int)

    # filter years
    long_df = filter_data(long_df, 'year', YEARS)

    return long_df
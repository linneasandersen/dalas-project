import pandas as pd


def merge_temp_change(primary_df, temp_df):
    """
    Merge temperature change data for both exporter and importer countries
    into the primary trade dataframe.
    """

    # Merge exporter temp change
    merged = primary_df.merge(
        temp_df.rename(columns={
            'country': 'exporter_name',
            'temp_change': 'exporter_country_temp_change',
            'std_dev_temp_change': 'exporter_country_temp_change_std'
        }),
        on=['exporter_name', 'year'],
        how='left'
    )

    # Merge importer temp change
    merged = merged.merge(
        temp_df.rename(columns={
            'country': 'importer_name',
            'temp_change': 'importer_country_temp_change',
            'std_dev_temp_change': 'importer_country_temp_change_std'
        }),
        on=['importer_name', 'year'],
        how='left'
    )

    return merged


def merge_fao_hs2cpc(fao_df, hs2cpc_df):
    return fao_df.merge(
        hs2cpc_df[['hs2', 'cpc product code']],
        on='hs2',
        how='left'
    )


def merge_gdp(trade_df, gdp_df):
    return trade_df.merge(
        gdp_df[['country name', 'year', 'gdp']].rename(columns={
            'country name': 'exporter_name',
            'gdp': 'exporter_gdp'
        }),
        on=['exporter_name', 'year'],
        how='left'
    ).merge(
        gdp_df[['country name', 'year', 'gdp']].rename(columns={
            'country name': 'importer_name',
            'gdp': 'importer_gdp'
        }),
        on=['importer_name', 'year'],
        how='left'
    )

def merge_land(trade_df, land_df):
    return trade_df.merge(
        land_df[['country name', 'year', 'land_area']].rename(columns={
            'country name': 'exporter_name',
            'land_area': 'exporter_land_area'
        }),
        on=['exporter_name', 'year'],
        how='left'
    ).merge(
        land_df[['country name', 'year', 'land_area']].rename(columns={
            'country name': 'importer_name',
            'land_area': 'importer_land_area'
        }),
        on=['importer_name', 'year'],
        how='left'
    )

def merge_population(trade_df, population_df):
    return trade_df.merge(
        population_df[['country name', 'year', 'population']].rename(columns={
            'country name': 'exporter_name',
            'population': 'exporter_population'
        }),
        on=['exporter_name', 'year'],
        how='left'
    ).merge(
        population_df[['country name', 'year', 'population']].rename(columns={
            'country name': 'importer_name',
            'population': 'importer_population'
        }),
        on=['importer_name', 'year'],
        how='left'
    )
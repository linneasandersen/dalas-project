import pandas as pd


def merge_temp_change(primary_df, temp_df):
    """
    Merge temperature change data for both exporter and importer countries
    into the primary trade dataframe.
    """
    # if importer_country_temp_change and exporter_country_temp_change columns already exist, delete them
    for col in ['exporter_country_temp_change', 'exporter_country_temp_change_std',
                'importer_country_temp_change', 'importer_country_temp_change_std']:
        if col in primary_df.columns:
            primary_df = primary_df.drop(columns=[col])

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

def merge_lat_long(trade_df, lat_long_df):
    return trade_df.merge(
        lat_long_df[['country name', 'latitude', 'longitude']].rename(columns={
            'country name': 'exporter_name',
            'latitude': 'exporter_latitude',
            'longitude': 'exporter_longitude'
        }),
        on='exporter_name',
        how='left'
    ).merge(
        lat_long_df[['country name', 'latitude', 'longitude']].rename(columns={
            'country name': 'importer_name',
            'latitude': 'importer_latitude',
            'longitude': 'importer_longitude'
        }),
        on='importer_name',
        how='left'
    )

def merge_logistics_index_old(trade_df, logistics_df):
    return trade_df.merge(
        logistics_df[['country name', 'year', 'logistics_index']].rename(columns={
            'country name': 'exporter_name',
            'logistics_index': 'exporter_logistics_index'
        }),
        on=['exporter_name', 'year'],
        how='left'
    ).merge(
        logistics_df[['country name', 'year', 'logistics_index']].rename(columns={
            'country name': 'importer_name',
            'logistics_index': 'importer_logistics_index'
        }),
        on=['importer_name', 'year'],
        how='left'
    )

def merge_logistics_index(trade_df, logistics_df):    
    # Keep only the necessary columns
    logistics_subset = logistics_df[['country name', 'year', 'logistics_index']].rename(
        columns={'country name': 'country', 'logistics_index': 'logistics_index'}
    )
    
    # Merge exporter logistics
    trade_df = trade_df.merge(
        logistics_subset.rename(columns={'country': 'exporter_name', 'logistics_index': 'exporter_logistics_index'}),
        on=['exporter_name', 'year'],
        how='left'
    )
    
    # Merge importer logistics
    trade_df = trade_df.merge(
        logistics_subset.rename(columns={'country': 'importer_name', 'logistics_index': 'importer_logistics_index'}),
        on=['importer_name', 'year'],
        how='left'
    )
    
    return trade_df



def merge_rta(trade_df, rta_df):
    rta_cols = ['exporter_name', 'importer_name', 'year', 
                'rta', 'cu', 'fta', 'psa', 'eia', 'cueia', 'ftaeia', 'psaeia']
    
    return trade_df.merge(
        rta_df[rta_cols],
        on=['exporter_name', 'importer_name', 'year'],
        how='left'
    )

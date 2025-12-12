
def rename_columns_with_units(df):
    rename_dict = {
        'value': 'trade_value_usd',
        'quantity': 'trade_quantity',
        'exporter_country_temp_change': 'exporter_temp_change_c',
        'exporter_country_temp_change_std': 'exporter_temp_change_std_c',
        'importer_country_temp_change': 'importer_temp_change_c',
        'importer_country_temp_change_std': 'importer_temp_change_std_c',
        'exporter_gdp': 'exporter_gdp_usd',
        'importer_gdp': 'importer_gdp_usd',
        'exporter_land_area': 'exporter_land_area_km2',
        'importer_land_area': 'importer_land_area_km2',
        'country_distance_km': 'countries_distance_km',
        'exporter_gdp_per_capita': 'exporter_gdp_per_capita_usd',
        'importer_gdp_per_capita': 'importer_gdp_per_capita_usd'
    }
    df = df.rename(columns=rename_dict)

    # add a dummy column for tariff rates if not present
    if 'tariff_rate_percent' not in df.columns:
        df['tariff_rate_percent'] = None
    
    return df
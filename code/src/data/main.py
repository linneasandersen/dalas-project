import pandas as pd
from src.config import YEARS, GOOGLE_DRIVE, MERGED_DIR, COUNTRIES_REGIONS
from src.processing.clean import clean_all
from src.processing.hs import update_config
from src.processing.pipelines import process_dataframe

from src.fetch.hs import fetch_hs, fetch_hs2cpc_mapping
from src.fetch.countries import fetch_countries
from src.fetch.oec import (
    delete_oec_trade_file, fetch_oec_trade, 
    fetch_oec_trade_all, fetch_oec_trade_file
)
from src.fetch.fao import fetch_FAO
from src.utils.io import save_df

from src.data.postprocessing import interpolate_logistics_index

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor

from src.data.loaders import (
    load_forest_data,
    load_logistics_index_data,
    load_production_data,
    load_temp_change_data,
    load_gdp,
    load_land_area,
    load_population_data,
    load_lat_long_data,
    load_rta_data,
    load_merged_data
)

from src.data.mergers import (
    merge_gdp,
    merge_logistics_index,
    merge_temp_change,
    merge_fao_hs2cpc,
    merge_land,
    merge_population,
    merge_lat_long,
    merge_rta,
)

from src.data.features import (
    encode_hs2,
    feature_distance_from_lat_long,
    feature_political_events,
    feature_trade_per_gdp,
    engineer_lagged_features,
    feature_regions_from_countries,
    feature_product_category,
    feature_gdp_per_capita,
    feature_same_region,
    feature_relative_population,
    feature_relative_gdp,
    feature_logistics_gap,
    feature_any_trade_agreement,
    feature_top10_percent_trade,
    feature_trade_volatility,
)

from src.models.baseline import baseline_model_OLS


# ---------------------------------------------------
# 1. Pipeline step functions
# ---------------------------------------------------

def build_hs_dataset():
    hs = fetch_hs()
    cleaned = clean_all([hs])[0]
    processed = process_dataframe("hs", cleaned)
    save_df(processed, "hs2_filter_products")
    return processed


def build_fao_dataset():
    forest, production, temp_change = fetch_FAO()
    # optionally clean/process here too
    return forest, production, temp_change


def build_oec_for_years(years, output_dir):
    dfs = []
    for year in years:
        print(f"Fetching OEC for {year}")
        oec = fetch_oec_trade_file(year, output_dir, filename=f"oec_trade_filtered_{year}.csv")

        cleaned = clean_all([oec])[0]
        #processed = process_dataframe("oec", cleaned)

        #save_df(processed, f"oec_trade_filtered_{year}", output_dir)
        dfs.append(cleaned)

        delete_oec_trade_file(year)

    combined = pd.concat(dfs, ignore_index=True)
    save_df(combined, "oec_trade_filtered_all_years", output_dir)
    return combined


def load_combined_oec(output_dir):
    return fetch_oec_trade_all(dir=output_dir, filename="oec_trade_filtered_all_years.csv")

def describe_df(df):
    print("Data Description:")
    print(df['value'].describe(include='all'))
    print("\nData Info:")
    print(df.info())
    print("\nData Columns:")
    print(df.columns)
    print("\nImporter Counts:")
    print(df['importer_name'].value_counts())
    print("\nExporter Counts:")
    print(df['exporter_name'].value_counts())


# ---------------------------------------------------
# 2. Configure which steps should run
# ---------------------------------------------------

PIPELINE_CONFIG = {
    "build_hs": False,
    "build_fao": False,
    "load_forest": False,
    "load_production": False,
    "load_gdp": False,
    "load_land": False,
    "load_pop": False,
    "load_lat_long": False,
    "load_temp_change": False,
    "load_rta": False,
    "load_logistics_index": False,
    #"load_lang": False,
    "build_oec_by_year": False,
    "load_combined_oec": False,
    "hs2cpc_mapping": False,
    "describe_oec": False,
    "merge_temp_change": False,
    "merge_gdp": False,
    "merge_land": False,
    "merge_pop": False,
    "merge_lat_long": False,
    "merge_logistics_index": False,
    "merge_rta": False,
    #"merge_lang": False,
    "engineer_features_regions": False,
    "engineer_features_distance_prodcat_gdppercap": False,
    "engineer_features_trade_per_gdp": False,
    "engineer_dummy_and_ratio_features": False,
    "engineer_features_volatility_top10pct": False,
    "engineer_features_same_region": False,
    "engineer_features_political": False,
    "engineer_lagged_features": False,
    "train_baseline": False,
    "rename_columns": False,
    "interpolate_logistics_index": False,
    "explore_latest_data": False,
    "return": True
}

# ---------------------------------------------------
# 3. Main orchestration
# ---------------------------------------------------

def build():
    processed_dir = GOOGLE_DRIVE / "processed"
    return_df = None

    if PIPELINE_CONFIG["return"]:
        return_df = load_merged_data(load_latest=True)

    if PIPELINE_CONFIG["explore_latest_data"]:
        df = load_merged_data(load_latest=True)  # test loading latest merged data
        #print(df.describe(include='all'))
        print(df.info())


    if PIPELINE_CONFIG["interpolate_logistics_index"]:
        df = load_merged_data(load_latest=True)
        print("Before interpolation:")
        print(df.info())
        df_interpolated = interpolate_logistics_index(df)
        print("After interpolation:")
        print(df_interpolated.info())
        save_df(df_interpolated, "oec_data_all_features_renamed_interpolated_logistics", MERGED_DIR)

    if PIPELINE_CONFIG["build_hs"]:
        hs = build_hs_dataset()
        print(hs.head())

    if PIPELINE_CONFIG["build_fao"]:
        forest, production, temp_change = build_fao_dataset()
        print(forest.head())

    if PIPELINE_CONFIG["load_forest"]:
        forest = load_forest_data()
        print(forest.head())
    
    if PIPELINE_CONFIG["load_production"]:
        production = load_production_data()
        print(production.head())

    if PIPELINE_CONFIG["load_gdp"]:
        gdp = load_gdp()
        print(gdp.head())

    if PIPELINE_CONFIG["load_land"]:
        land = load_land_area()
        print(land.head())

    if PIPELINE_CONFIG["load_pop"]:
        pop = load_population_data()
        print(pop.head())

    if PIPELINE_CONFIG["load_lat_long"]:
        lat_long = load_lat_long_data()
        print(lat_long.head())
    
    if PIPELINE_CONFIG["load_temp_change"]:
        temp_change = load_temp_change_data()
        print(temp_change.head())

    if PIPELINE_CONFIG["load_logistics_index"]:
        logistics_index = load_logistics_index_data()
        print(logistics_index.head())

    if PIPELINE_CONFIG["load_rta"]:
        rta = load_rta_data()
        print(rta.head())
        
    if PIPELINE_CONFIG["build_oec_by_year"]:
        oec_all = build_oec_for_years(YEARS, processed_dir)
        print(oec_all.head())

    if PIPELINE_CONFIG["load_combined_oec"]:
        oec_all = load_combined_oec(processed_dir)
        print(oec_all.head())

    if PIPELINE_CONFIG["hs2cpc_mapping"]:
        hs2cpc = fetch_hs2cpc_mapping()
        print(hs2cpc.head())
        print(hs2cpc.columns)
        clean_hs2cpc = clean_all([hs2cpc])[0]
        hs2cpc = process_dataframe("hs2cpc_mapping", clean_hs2cpc)
        save_df(hs2cpc, "hs6cpc_mapping", processed_dir)
        print(hs2cpc.head())
        print(hs2cpc['cpc product code'].unique().tolist())

    if PIPELINE_CONFIG["merge_temp_change"]:
        MERGED_DIR.mkdir(exist_ok=True)
        oec_all = load_merged_data(load_latest=True)
        temp_change = load_temp_change_data()
        merged_df = merge_temp_change(oec_all, temp_change)
        merged_df['exporter_name'] = merged_df['exporter_name'].astype('string')
        merged_df['importer_name'] = merged_df['importer_name'].astype('string')
        save_df(merged_df, "data_remerge_temp_change", MERGED_DIR)
        print(merged_df.columns)

    if PIPELINE_CONFIG["merge_gdp"]:
        MERGED_DIR.mkdir(exist_ok=True)
        oec_all = load_merged_data()
        print("OEC data:")
        print(oec_all.head())
        gdp = load_gdp()
        merged_df = merge_gdp(oec_all, gdp)
        #save_df(merged_df, "oec_trade_temp_change_gdp", MERGED_DIR)
        print("Merged data with GDP:")
        print(merged_df.head().to_string()) 
        print(merged_df.columns)

    if PIPELINE_CONFIG["merge_land"]:
        oec_all = load_merged_data(load_latest=True)
        print("OEC data:")
        print(oec_all.head())
        land = load_land_area()
        merged_df = merge_land(oec_all, land)
        save_df(merged_df, "oec_trade_temp_change_gdp_regions_land", MERGED_DIR)
        print("Merged data with land:")
        print(merged_df.head().to_string()) 
        print(merged_df.columns)
    
    if PIPELINE_CONFIG["merge_pop"]:
        oec_all = load_merged_data(load_latest=True)
        print("OEC data:")
        print(oec_all.head())
        pop = load_population_data()
        merged_df = merge_population(oec_all, pop)
        save_df(merged_df, "oec_trade_temp_change_gdp_regions_land_pop", MERGED_DIR)
        print("Merged data with population:")
        print(merged_df.head().to_string()) 
        print(merged_df.columns)

    if PIPELINE_CONFIG["merge_lat_long"]:
        oec_all = load_merged_data(load_latest=True)
        print("OEC data:")
        print(oec_all.head())
        lat_long = load_lat_long_data()
        merged_df = merge_lat_long(oec_all, lat_long)
        save_df(merged_df, "oec_trade_temp_change_gdp_regions_land_pop_latlong", MERGED_DIR)
        print("Merged data with lat/long:")
        #print(merged_df.head().to_string()) 
        print(merged_df.columns)
        print(merged_df[['exporter_name', 'exporter_latitude', 'exporter_longitude', 'importer_name', 'importer_latitude', 'importer_longitude']].head().to_string())
        
    if PIPELINE_CONFIG["merge_logistics_index"]:
        oec_all = load_merged_data(load_latest=True)
        print("OEC data:")
        print(oec_all.head())
        logistics_index = load_logistics_index_data()
        merged_df = merge_logistics_index(oec_all, logistics_index)
        save_df(merged_df, "oec_data_all_features_renamed", MERGED_DIR)
        print("Merged data with logistics index:")
        print(merged_df.columns)
        print(merged_df[['exporter_name', 'exporter_logistics_index', 'importer_name', 'importer_logistics_index']].head().to_string())

    if PIPELINE_CONFIG["merge_rta"]:
        oec_all = load_merged_data(load_latest=True)
        print("OEC data:")
        print(oec_all.head())
        rta = load_rta_data()
        merged_df = merge_rta(oec_all, rta)
        save_df(merged_df, "oec_trade_temp_change_gdp_land_pop_latlong_logistics_rta_with_features_renamed_test", MERGED_DIR)
        print("Merged data with logistics index:")
        print(merged_df.columns)


    if PIPELINE_CONFIG["engineer_features_regions"]:
        df = load_merged_data(load_latest=True)
        print(df.columns)
        merged_df_regions = feature_regions_from_countries(df, 'exporter_name', COUNTRIES_REGIONS, 'exporter_region')
        merged_df_regions = feature_regions_from_countries(merged_df_regions, 'importer_name', COUNTRIES_REGIONS, 'importer_region')
        print(merged_df_regions.head().to_string()) 
        print(merged_df_regions.columns)
        save_df(merged_df_regions, "oec_trade_temp_change_gdp_regions", MERGED_DIR)

    if PIPELINE_CONFIG["engineer_features_distance_prodcat_gdppercap"]:
        df = load_merged_data(load_latest=True)
        feature_df = feature_distance_from_lat_long(df)
        feature_df = feature_product_category(feature_df)
        feature_df = feature_gdp_per_capita(feature_df)
        print(feature_df.head().to_string()) 
        print(feature_df.columns)
        save_df(feature_df, "oec_trade_temp_change_gdp_regions_land_pop_latlong_distance", MERGED_DIR)

    if PIPELINE_CONFIG["engineer_features_trade_per_gdp"]:
        df = load_merged_data(load_latest=True)
        feature_df = feature_trade_per_gdp(df)
        print(feature_df.head().to_string()) 
        print(feature_df.columns)
        save_df(feature_df, "oec_trade_temp_change_gdp_land_pop_latlong_logistics_with_features_renamed", MERGED_DIR)
    
    if PIPELINE_CONFIG["engineer_dummy_and_ratio_features"]:
        df = load_merged_data(load_latest=True)
        feature_df = feature_any_trade_agreement(df)
        feature_df = feature_relative_population(feature_df)
        feature_df = feature_relative_gdp(feature_df)
        feature_df = feature_logistics_gap(feature_df)
        print(feature_df.head().to_string()) 
        print(feature_df.columns)
        save_df(feature_df, "data_more_features", MERGED_DIR)
    
    if PIPELINE_CONFIG["engineer_features_volatility_top10pct"]:
        df = load_merged_data(load_latest=True)
        feature_df = feature_top10_percent_trade(df)
        feature_df = feature_trade_volatility(feature_df)
        print(feature_df.head().to_string()) 
        print(feature_df.columns)
        save_df(feature_df, "data_more_features2", MERGED_DIR)

    if PIPELINE_CONFIG["engineer_features_same_region"]:
        df = load_merged_data(load_latest=True)
        feature_df = feature_same_region(df)
        print(feature_df.head().to_string()) 
        print(feature_df.columns)
        save_df(feature_df, "data_more_features3", MERGED_DIR)
    
    if PIPELINE_CONFIG["engineer_features_political"]:
        df = load_merged_data(load_latest=True)
        feature_df = feature_political_events(df)
        print(feature_df.head().to_string()) 
        print(feature_df.columns)
        save_df(feature_df, "data_more_features4", MERGED_DIR)

    if PIPELINE_CONFIG["rename_columns"]:
        from src.data.postprocessing import rename_columns_with_units
        df = load_merged_data(load_latest=True)
        renamed_df = rename_columns_with_units(df)
        print(renamed_df.head().to_string()) 
        print(renamed_df.columns)
        save_df(renamed_df, "data_remerged_renames", MERGED_DIR)

    return return_df


# TODO: clean other FAO datasets and process them
# TODO: merge FAO datasets with primary dataset
# TODO: merge with tariff data
# TODO: try baseline model with new columns
# TODO: try baseline model with per product dataset


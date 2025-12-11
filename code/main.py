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

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor

from src.data.loaders import (
    load_forest_data,
    load_production_data,
    load_temp_change_data,
    load_gdp,
    load_land_area,
    load_population_data,
)

from src.data.mergers import (
    merge_gdp,
    merge_temp_change,
    merge_fao_hs2cpc,
    merge_land,
    merge_population
)

from src.data.features import (
    encode_hs2,
    panel_train_val_test_split,
    rolling_panel_split,
    engineer_lagged_features,
    feature_regions_from_countries
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

def load_merged_data(load_latest=False):
    merged_dir = GOOGLE_DRIVE / "processed" / "merged"
    if load_latest:
        # find most recent file in directory
        import os
        files = os.listdir(merged_dir)
        csv_files = [f for f in files if f.endswith('.csv')]
        latest_file = max(csv_files, key=lambda x: os.path.getctime(os.path.join(merged_dir, x)))
        merged_df = pd.read_csv(merged_dir / latest_file)
    else:
        merged_df = pd.read_csv(merged_dir / "oec_trade_with_temp_change.csv")
    return merged_df

def load_lagged_data():
    merged_dir = GOOGLE_DRIVE / "processed" / "merged"
    lagged_df = pd.read_csv(merged_dir / "oec_trade_lagged.csv")
    return lagged_df


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
    "load_temp_change": False,
    "build_oec_by_year": False,
    "load_combined_oec": False,
    "hs2cpc_mapping": False,
    "describe_oec": False,
    "merge_temp_change": False,
    "merge_gdp": False,
    "merge_land": False,
    "merge_pop": True,
    "engineer_features": False,
    "engineer_lagged_features": False,
    "train_baseline": False,
}

# ---------------------------------------------------
# 3. Main orchestration
# ---------------------------------------------------

def main():
    processed_dir = GOOGLE_DRIVE / "processed"

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
    
    if PIPELINE_CONFIG["load_temp_change"]:
        temp_change = load_temp_change_data()
        print(temp_change.head())

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
        oec_all = load_combined_oec(processed_dir)
        temp_change = load_temp_change_data()
        merged_df = merge_temp_change(oec_all, temp_change)
        merged_df['exporter_name'] = merged_df['exporter_name'].astype('string')
        merged_df['importer_name'] = merged_df['importer_name'].astype('string')
        save_df(merged_df, "oec_trade_with_temp_change", MERGED_DIR)
        describe_df(merged_df)
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

    if PIPELINE_CONFIG["engineer_features"]:
        df = load_merged_data(load_latest=True)
        print(df.columns)
        merged_df_regions = feature_regions_from_countries(df, 'exporter_name', COUNTRIES_REGIONS, 'exporter_region')
        merged_df_regions = feature_regions_from_countries(merged_df_regions, 'importer_name', COUNTRIES_REGIONS, 'importer_region')
        print(merged_df_regions.head().to_string()) 
        print(merged_df_regions.columns)
        save_df(merged_df_regions, "oec_trade_temp_change_gdp_regions", MERGED_DIR)

    if PIPELINE_CONFIG["engineer_lagged_features"]:
        merged_df = load_merged_data()
        feature_cols = [
                        'exporter_country_temp_change',
                        'importer_country_temp_change', 
                        'value'
                        ]
        merged_df = engineer_lagged_features(
            merged_df,
            group_cols=['exporter_id', 'importer_id', 'hs2'],
            time_col='year',
            feature_cols=feature_cols,
            lags=[1,2,3]
        )
        save_df(merged_df, "oec_trade_lagged", MERGED_DIR)
        print(merged_df.head())
        print(merged_df.columns)

    if PIPELINE_CONFIG["train_baseline"]:
    # Load merged lagged data
        merged_df = load_lagged_data()

        # Step 1: One-hot encode hs2 
        primary_df_encoded = encode_hs2(merged_df, hs_col='hs2')

        # Step 2: Define features
        feature_cols = [
            #'quantity',   # optionally include
            'exporter_country_temp_change',
            #'exporter_country_temp_change_std',
            'importer_country_temp_change',
            #'importer_country_temp_change_std',
            'exporter_country_temp_change_lag_1',
            'exporter_country_temp_change_lag_2',
            'exporter_country_temp_change_lag_3',
            'importer_country_temp_change_lag_1',
            'importer_country_temp_change_lag_2',
            'importer_country_temp_change_lag_3',
            'value_lag_1',
            'value_lag_2',
            'value_lag_3'
        ] + [c for c in primary_df_encoded.columns if c.startswith('hs2_')]  # include hs2 dummies

        # Step 3: Train/val/test split
        train_df, val_df, test_df = panel_train_val_test_split(primary_df_encoded)

        # Step 4: Train baseline OLS
        model, val_rmse = baseline_model_OLS(train_df, val_df, feature_cols, target_col='value')

        print("Val set RMSE:")
        print(val_rmse)


if __name__ == "__main__":
    main()



# TODO: clean other FAO datasets and process them
# TODO: merge FAO datasets with primary dataset
# TODO: merge with tariff data
# TODO: try baseline model with new columns
# TODO: try baseline model with per product dataset


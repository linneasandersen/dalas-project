import pandas as pd
from src.config import YEARS, GOOGLE_DRIVE
from src.processing.clean import clean_all
from src.processing.hs import update_config
from src.processing.pipelines import process_dataframe

from src.fetch.hs import fetch_hs
from src.fetch.countries import fetch_countries
from src.fetch.oec import (
    delete_oec_trade_file, fetch_oec_trade, 
    fetch_oec_trade_all, fetch_oec_trade_file
)
from src.fetch.fao import fetch_FAO
from src.utils.io import save_df


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

def load_forest_data():
    """Load, clean, and process FAO forest dataset."""
    forest = pd.read_csv(GOOGLE_DRIVE / "processed" / "FAO_forest_data.csv")
    forest_cleaned = clean_all([forest])[0]
    forest_processed = process_dataframe("fao_forest", forest_cleaned)
    return forest_processed


def load_production_data():
    """Load, clean, and process FAO production dataset."""
    production = pd.read_csv(GOOGLE_DRIVE / "processed" / "FAO_prod_data.csv")
    production_cleaned = clean_all([production])[0]
    production_processed = process_dataframe("fao_production", production_cleaned)
    return production_processed


def load_temp_change_data():
    """Load, clean, and process FAO temperature change dataset."""
    temp_change = pd.read_csv(GOOGLE_DRIVE / "processed" / "FAO_env_data.csv")
    temp_change_cleaned = clean_all([temp_change])[0]
    temp_change_processed = process_dataframe("fao_temp_change", temp_change_cleaned)
    return temp_change_processed


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
    print(df.describe(include='all'))
    print("\nData Info:")
    print(df.info())
    print("\nData Columns:")
    print(df.columns)
    print("\nImporter Counts:")
    print(df['importer_name'].value_counts())
    print("\nExporter Counts:")
    print(df['exporter_name'].value_counts())

def merge_temp_change(primary_df, temp_df):
    """
    Merge temperature change data for both exporter and importer countries
    into the primary trade dataframe.

    Parameters
    ----------
    primary_df : pd.DataFrame
        Main trade dataframe with columns ['year', 'exporter_name', 'importer_name', ...]
    temp_df : pd.DataFrame
        Temperature change dataframe with columns ['country', 'year', 'temp_change', 'std_dev_temp_change']

    Returns
    -------
    pd.DataFrame
        Primary dataframe with new columns:
        'exporter_country_temp_change', 'exporter_country_temp_change_std',
        'importer_country_temp_change', 'importer_country_temp_change_std'
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



# ---------------------------------------------------
# 2. Configure which steps should run
# ---------------------------------------------------

PIPELINE_CONFIG = {
    "build_hs": False,
    "build_fao": False,
    "load_forest": False,
    "load_production": False,
    "load_temp_change": False,
    "build_oec_by_year": False,
    "load_combined_oec": False,
    "describe_oec": False,
    "merge_temp_change": True
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
    
    if PIPELINE_CONFIG["load_temp_change"]:
        temp_change = load_temp_change_data()
        print(temp_change.head())

    if PIPELINE_CONFIG["build_oec_by_year"]:
        oec_all = build_oec_for_years(YEARS, processed_dir)
        print(oec_all.head())

    if PIPELINE_CONFIG["load_combined_oec"]:
        oec_all = load_combined_oec(processed_dir)
        print(oec_all.head())


    if PIPELINE_CONFIG["merge_temp_change"]:
        merged_dir = processed_dir / "merged"
        merged_dir.mkdir(exist_ok=True)
        oec_all = load_combined_oec(processed_dir)
        temp_change = load_temp_change_data()
        merged_df = merge_temp_change(oec_all, temp_change)
        #save_df(merged_df, "oec_trade_with_temp_change", merged_dir)
        describe_df(merged_df)


if __name__ == "__main__":
    main()



# TODO: train simple model on merged data
# TODO: construct lagged features
# TODO: clean other FAO datasets and process them
# TODO: merge FAO datasets with primary dataset
# TODO: merge with tariff data

import pandas as pd
from src.config import GOOGLE_DRIVE, LAGGED_DIR
from src.processing.clean import clean_all
from src.processing.pipelines import process_dataframe
import os

# ---------------------------------------------------
# Generic helper
# ---------------------------------------------------

def load_csv_pipeline(path, dataset_name, **read_kwargs):
    df = pd.read_csv(path, **read_kwargs)
    df = clean_all([df])[0]
    df = process_dataframe(dataset_name, df)
    return df

# ---------------------------------------------------
# Specific loaders
# ---------------------------------------------------

def load_forest_data():
    return load_csv_pipeline(
        GOOGLE_DRIVE / "processed" / "FAO_forest_data.csv",
        "fao_forest"
    )

def load_production_data():
    return load_csv_pipeline(
        GOOGLE_DRIVE / "processed" / "FAO_prod_data.csv",
        "fao_production",
        index_col=0
    )

def load_temp_change_data():
    return load_csv_pipeline(
        GOOGLE_DRIVE / "processed" / "FAO_env_data.csv",
        "fao_temp_change"
    )

def load_gdp():
    return load_csv_pipeline(
        GOOGLE_DRIVE / "raw" / "Country_GDP.csv",
        "gdp",
        skiprows=4
    )

def load_land_area():
    return load_csv_pipeline(
        GOOGLE_DRIVE / "raw" / "Country_Land_Area.csv",
        "land",
        skiprows=4
    )

def load_population_data():
    return load_csv_pipeline(
        GOOGLE_DRIVE / "raw" / "Country_Population.csv",
        "population",
        skiprows=4
    )

def load_lat_long_data():
    return load_csv_pipeline(
        GOOGLE_DRIVE / "raw" / "countries.csv",
        "lat_long"
    )

def load_logistics_index_data():
    return load_csv_pipeline(
        GOOGLE_DRIVE / "raw" / "Logistics_Index.csv",
        "logistics_index",
        skiprows=4
    )

def load_rta_data():
    return load_csv_pipeline(
        GOOGLE_DRIVE / "raw" / "rta.csv",
        "rta"
    )

def load_merged_data(load_latest=False):
    merged_dir = GOOGLE_DRIVE / "processed" / "merged"
    if load_latest:
        # find most recent file in directory
        files = os.listdir(merged_dir)
        csv_files = [f for f in files if f.endswith('.csv')]
        latest_file = max(csv_files, key=lambda x: os.path.getctime(os.path.join(merged_dir, x)))
        print(f"Loading latest merged data file: {latest_file}")
        merged_df = pd.read_csv(
            merged_dir / latest_file,
            engine='python'
        )
    else:
        merged_df = pd.read_csv(merged_dir / "oec_trade_with_temp_change.csv")
    return merged_df

def load_latest_eda_data():
    eda_dir = GOOGLE_DRIVE / "processed" / "eda"
    # find most recent file in directory
    files = os.listdir(eda_dir)
    csv_files = [f for f in files if f.endswith('.csv')]
    latest_file = max(csv_files, key=lambda x: os.path.getctime(os.path.join(eda_dir, x)))
    print(f"Loading latest eda data file: {latest_file}")
    eda_df = pd.read_csv(
        eda_dir / latest_file,
        engine='python'
    )
    return eda_df


def load_lagged_data(number_of_lags: int) -> pd.DataFrame:
    filename = f"data_lagged_{number_of_lags}.csv"
    training_df = pd.read_csv(LAGGED_DIR / filename)
    return training_df
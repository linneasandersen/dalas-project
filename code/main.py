import pandas as pd
from src.config import YEARS, GOOGLE_DRIVE, MERGED_DIR
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
    production = pd.read_csv(GOOGLE_DRIVE / "processed" / "FAO_prod_data.csv", index_col=0)
    production_cleaned = clean_all([production])[0]
    production_processed = process_dataframe("fao_production", production_cleaned)
    return production_processed

def load_gdp():
    """Load, clean, and process FAO production dataset."""
    gdp = pd.read_csv(GOOGLE_DRIVE / "raw" / "Country_GDP.csv", skiprows=4)
    gdp_cleaned = clean_all([gdp])[0]
    gdp_processed = process_dataframe("gdp", gdp_cleaned)
    return gdp_processed


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
    print(df['value'].describe(include='all'))
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

def merge_fao_hs2cpc(fao_df, hs2cpc_df):
    merged_df = fao_df.merge(
        hs2cpc_df[['hs2', 'cpc product code']],
        on='hs2',
        how='left'
    )
    return merged_df


def encode_hs2(df, hs_col='hs2', drop_original=True):
    """
    One-hot encode the hs2 column in the dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the hs2 column.
    hs_col : str
        Name of the column to encode (default 'hs2')
    drop_original : bool
        Whether to drop the original hs2 column after encoding.
    
    Returns
    -------
    df_encoded : pd.DataFrame
        DataFrame with one-hot encoded hs2 columns added.
    """
    hs_dummies = pd.get_dummies(df[hs_col], prefix=hs_col)
    df_encoded = pd.concat([df, hs_dummies], axis=1)
    if drop_original:
        df_encoded = df_encoded.drop(columns=[hs_col])
    return df_encoded

def load_merged_data():
    merged_dir = GOOGLE_DRIVE / "processed" / "merged"
    merged_df = pd.read_csv(merged_dir / "oec_trade_with_temp_change.csv")
    return merged_df

def load_lagged_data():
    merged_dir = GOOGLE_DRIVE / "processed" / "merged"
    lagged_df = pd.read_csv(merged_dir / "oec_trade_lagged.csv")
    return lagged_df

def panel_train_val_test_split(df, time_col='year', last_test_year=2023, val_years=2):
    """
    Splits a panel dataset (country × product × time) into train, validation, and test sets.
    
    Parameters:
        df: pandas DataFrame containing columns for 'year', 'origin', 'product', 'destination', 'value'
        time_col: name of the time column
        last_test_year: year to reserve for testing
        val_years: number of years to use for validation (immediately before test year)
    
    Returns:
        train_df, val_df, test_df
    """
    
    # Ensure data is sorted by time
    df = df.sort_values(time_col).reset_index(drop=True)
    
    # Test set: the last year
    test_df = df[df[time_col] == last_test_year]
    
    # Validation set: val_years before test year
    val_years_range = range(last_test_year - val_years, last_test_year)
    val_df = df[df[time_col].isin(val_years_range)]
    
    # Training set: all remaining years
    train_df = df[~df[time_col].isin(list(val_years_range) + [last_test_year])]
    
    return train_df, val_df, test_df


def rolling_panel_split(df, time_col='year', last_test_year=2023, initial_train_years=10, val_window=1):
    """
    Creates rolling train/validation splits for a panel dataset while keeping the last year as test set.
    
    Parameters:
        df: pandas DataFrame with columns ['origin', 'product', 'destination', 'year', 'value']
        time_col: name of the time column
        last_test_year: year to reserve as final test set
        initial_train_years: number of earliest years to use for initial training window
        val_window: number of years in each validation window
    
    Returns:
        splits: list of (train_df, val_df) tuples for rolling validation
        test_df: DataFrame for the final test year
    """
    
    # Ensure data is sorted by time
    df = df.sort_values(time_col).reset_index(drop=True)
    
    # Test set: last year
    test_df = df[df[time_col] == last_test_year]
    
    # Keep only data before test year for rolling splits
    train_val_df = df[df[time_col] < last_test_year]
    
    # List of all years available for rolling
    years = sorted(train_val_df[time_col].unique())
    
    splits = []
    start_idx = 0
    while start_idx + initial_train_years + val_window <= len(years):
        train_years = years[start_idx:start_idx + initial_train_years]
        val_years = years[start_idx + initial_train_years : start_idx + initial_train_years + val_window]
        
        train_df = train_val_df[train_val_df[time_col].isin(train_years)]
        val_df = train_val_df[train_val_df[time_col].isin(val_years)]
        
        splits.append((train_df, val_df))
        start_idx += 1  # roll window by 1 year (can adjust)
    
    return splits, test_df

def engineer_lagged_features(df, group_cols, time_col='year', feature_cols=['value'], lags=[1,2,3]):
    """
    Engineer lagged features for panel data.
    
    Parameters:
        df: pandas DataFrame with panel data
        group_cols: list of columns to group by (e.g., ['origin', 'product', 'destination'])
        time_col: name of the time column
        feature_cols: list of feature columns to create lags for
        lags: list of integers indicating lag periods
    Returns:
        df: DataFrame with new lagged feature columns
    """
    df = df.sort_values(group_cols + [time_col])
    
    for feature in feature_cols:
        for lag in lags:
            lagged_col_name = f"{feature}_lag_{lag}"
            df[lagged_col_name] = df.groupby(group_cols)[feature].shift(lag)
    
    return df


def baseline_model_OLS_old(train_df, val_df, feature_cols, target_col='value', log_transform=True):
    """
    Train a baseline OLS model using lagged features, temperature, and optionally hs2.
    Only drops rows with NaNs in lagged features. Supports log-transform of target.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataframe.
    val_df : pd.DataFrame
        Validation dataframe.
    feature_cols : list
        List of columns to use as features.
    target_col : str
        Target column name (default 'value').
    log_transform : bool
        Whether to log-transform target and lagged features to stabilize scale.
    
    Returns
    -------
    pipeline : sklearn Pipeline
        Trained LinearRegression pipeline with scaler.
    rmse : float
        Validation RMSE (original scale).
    """
    
    # Identify lagged features
    lagged_features = [col for col in feature_cols if '_lag_' in col]
    
    print(f"Training before removing NaNs: {len(train_df)} rows")
    print(f"Validation before removing NaNs: {len(val_df)} rows")

    # Drop rows with NaNs in lagged features
    train_clean = train_df.dropna(subset=lagged_features).copy()
    val_clean = val_df.dropna(subset=lagged_features).copy()

    print(f"Training on {len(train_clean)} rows, validating on {len(val_clean)} rows after dropping NaNs in lagged features.")

    # Log-transform target and lagged features if requested
    if log_transform:
        train_clean[target_col + '_log'] = np.log1p(train_clean[target_col])
        val_clean[target_col + '_log'] = np.log1p(val_clean[target_col])
        
        # log-transform lagged features
        for f in lagged_features:
            train_clean[f + '_log'] = np.log1p(train_clean[f])
            val_clean[f + '_log'] = np.log1p(val_clean[f])
        
        # Replace lagged features in feature_cols with log versions
        feature_cols_log = [f + '_log' if f in lagged_features else f for f in feature_cols]
    else:
        feature_cols_log = feature_cols

    # Build pipeline with scaler + OLS
    pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ])
    
    # Train
    pipeline.fit(train_clean[feature_cols_log], train_clean[target_col + ('_log' if log_transform else '')])
    
    # Predict
    val_preds_log = pipeline.predict(val_clean[feature_cols_log])
    
    # Convert back to original scale if log-transform was applied
    if log_transform:
        val_preds = np.expm1(val_preds_log)
    else:
        val_preds = val_preds_log
    
    # Compute RMSE on original scale
    rmse = np.sqrt(mean_squared_error(val_clean[target_col], val_preds))
    rmse_log = np.sqrt(mean_squared_error(
        val_clean[target_col + "_log"],
        val_preds_log
    ))

    print("hello")
    #print(f"Validation RMSE: {rmse_log}")
    print(f"Validation RMSE (log scale): {rmse_log}")
    
    return pipeline, rmse

def compute_mape_smape(y_true, y_pred):
    """
    Compute MAPE and SMAPE between true and predicted values.
    
    Parameters
    ----------
    y_true : array-like
        True target values (original scale)
    y_pred : array-like
        Predicted values (original scale)
        
    Returns
    -------
    mape : float
        Mean Absolute Percentage Error
    smape : float
        Symmetric Mean Absolute Percentage Error
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Avoid division by zero for MAPE
    nonzero_mask = y_true != 0
    mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
    
    # SMAPE formula
    smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100
    
    return mape, smape



def baseline_model_OLS(train_df, val_df, feature_cols, target_col='value', log_transform=True):
    """
    Train a baseline OLS model using lagged features, temperature, and optionally hs2.
    Only drops rows with NaNs in lagged features. Supports log-transform of target.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataframe.
    val_df : pd.DataFrame
        Validation dataframe.
    feature_cols : list
        List of columns to use as features.
    target_col : str
        Target column name (default 'value').
    log_transform : bool
        Whether to log-transform target and lagged features to stabilize scale.

    Returns
    -------
    pipeline : sklearn Pipeline
        Trained LinearRegression pipeline with scaler.
    rmse : float
        Validation RMSE (original scale).
    """

    # Identify lagged features
    lagged_features = [col for col in feature_cols if '_lag_' in col]

    print(f"Training before removing NaNs: {len(train_df)} rows")
    print(f"Validation before removing NaNs: {len(val_df)} rows")

    # Drop rows with NaNs in lagged features
    train_clean = train_df.dropna(subset=lagged_features).copy()
    val_clean = val_df.dropna(subset=lagged_features).copy()

    print(f"Training on {len(train_clean)} rows, validating on {len(val_clean)} rows after dropping NaNs in lagged features.")

    # Log-transform target and lagged features if requested
    if log_transform:
        # Target
        train_clean.loc[:, target_col + '_log'] = np.log1p(train_clean[target_col])
        val_clean.loc[:, target_col + '_log'] = np.log1p(val_clean[target_col])

        # Lagged features
        for f in lagged_features:
            # Clip at 0 to avoid negative values for log1p
            train_clean.loc[:, f + '_log'] = np.log1p(train_clean[f].clip(lower=0))
            val_clean.loc[:, f + '_log'] = np.log1p(val_clean[f].clip(lower=0))

        # Replace lagged features in feature_cols with log versions
        feature_cols_log = [f + '_log' if f in lagged_features else f for f in feature_cols]
        target_col_used = target_col + '_log'
    else:
        feature_cols_log = feature_cols
        target_col_used = target_col

    # Build pipeline with scaler + OLS
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])

    # Train
    pipeline.fit(train_clean[feature_cols_log], train_clean[target_col_used])

    # Predict
    val_preds_log = pipeline.predict(val_clean[feature_cols_log])

    # Convert back to original scale if log-transform was applied
    if log_transform:
        val_preds = np.expm1(val_preds_log)
    else:
        val_preds = val_preds_log

    # Compute RMSE on original scale
    rmse = np.sqrt(mean_squared_error(val_clean[target_col], val_preds))
    rmse_log = np.sqrt(mean_squared_error(
        val_clean[target_col + "_log"],
        val_preds_log
    ))
    mape, smape = compute_mape_smape(val_clean[target_col], val_preds)
    print(f"MAPE: {mape:.2f}%")
    print(f"SMAPE: {smape:.2f}%")

    print(f"Validation RMSE: {rmse}")
    print(f"Validation RMSE (log scale): {rmse_log}")
    
    factor_error = np.expm1(rmse_log) + 1  # optional +1 if you used log1p
    print(f"On average, predictions are within a factor of ~{factor_error:.2f} of the true values")

    return pipeline, rmse


# ---------------------------------------------------
# 2. Configure which steps should run
# ---------------------------------------------------

PIPELINE_CONFIG = {
    "build_hs": False,
    "build_fao": False,
    "load_forest": False,
    "load_production": False,
    "load_gdp": True,
    "load_temp_change": False,
    "build_oec_by_year": False,
    "load_combined_oec": False,
    "hs2cpc_mapping": False,
    "describe_oec": False,
    "merge_temp_change": False,
    "engineer_features": False,
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

    if PIPELINE_CONFIG["engineer_features"]:
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


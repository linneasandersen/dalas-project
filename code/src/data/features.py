import numpy as np
import pandas as pd
from src.config import PRODUCT_CATEGORIES


# ---------------------------------------------------
# Encoding
# ---------------------------------------------------

def encode_hs2(df, hs_col='hs2', drop_original=True):
    hs_dummies = pd.get_dummies(df[hs_col], prefix=hs_col)
    df_encoded = pd.concat([df, hs_dummies], axis=1)

    if drop_original:
        df_encoded = df_encoded.drop(columns=[hs_col])

    return df_encoded


# ---------------------------------------------------
# Panel splits
# ---------------------------------------------------

def panel_train_val_test_split(df, time_col='year', last_test_year=2023, val_years=2):
    df = df.sort_values(time_col).reset_index(drop=True)

    test_df = df[df[time_col] == last_test_year]

    val_years_range = range(last_test_year - val_years, last_test_year)
    val_df = df[df[time_col].isin(val_years_range)]

    train_df = df[~df[time_col].isin(list(val_years_range) + [last_test_year])]

    return train_df, val_df, test_df


def rolling_panel_split(df, time_col='year', last_test_year=2023, initial_train_years=10, val_window=1):
    df = df.sort_values(time_col).reset_index(drop=True)

    test_df = df[df[time_col] == last_test_year]
    train_val_df = df[df[time_col] < last_test_year]

    years = sorted(train_val_df[time_col].unique())

    splits = []
    start_idx = 0
    while start_idx + initial_train_years + val_window <= len(years):
        train_years = years[start_idx:start_idx + initial_train_years]
        val_years = years[start_idx + initial_train_years : start_idx + initial_train_years + val_window]

        train_df = train_val_df[train_val_df[time_col].isin(train_years)]
        val_df = train_val_df[train_val_df[time_col].isin(val_years)]

        splits.append((train_df, val_df))
        start_idx += 1

    return splits, test_df


# ---------------------------------------------------
# Lagged features
# ---------------------------------------------------

def engineer_lagged_features(df, group_cols, time_col='year', feature_cols=['value'], lags=[1,2,3]):
    df = df.sort_values(group_cols + [time_col])

    for feature in feature_cols:
        for lag in lags:
            lagged_col_name = f"{feature}_lag_{lag}"
            df[lagged_col_name] = df.groupby(group_cols)[feature].shift(lag)

    return df

# ---------------------------------------------------
# Other features
# ---------------------------------------------------

def feature_regions_from_countries(df, country_col='country name', region_map={}, column_name='region'):
    df[column_name] = df[country_col].map(region_map)
    return df

def feature_distance_from_lat_long(df):
    from geopy.distance import geodesic

    def calculate_distance(row):
        exporter_coords = (row['exporter_latitude'], row['exporter_longitude'])
        importer_coords = (row['importer_latitude'], row['importer_longitude'])
        return geodesic(exporter_coords, importer_coords).kilometers

    df['country_distance_km'] = df.apply(calculate_distance, axis=1)
    return df

def feature_product_category(df, hs_col='hs2', category_map=PRODUCT_CATEGORIES):
    df['product_category'] = df[hs_col].map(category_map)
    return df

def feature_gdp_per_capita(df):
    df['exporter_gdp_per_capita'] = df['exporter_gdp'] / df['exporter_population']
    df['importer_gdp_per_capita'] = df['importer_gdp'] / df['importer_population']
    return df

def feature_trade_per_gdp(df):
    df['trade_value_per_exporter_gdp_usd'] = df['trade_value_usd'] / df['exporter_gdp_usd']
    df['trade_value_per_importer_gdp_usd'] = df['trade_value_usd'] / df['importer_gdp_usd']
    return df

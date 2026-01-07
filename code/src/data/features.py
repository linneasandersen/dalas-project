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
# Lagged features
# ---------------------------------------------------

def engineer_lagged_features(df, group_cols, time_col='year', feature_cols=['value'], lags=[1, 2, 3]):
    df = df.sort_values(group_cols + [time_col])

    for feature in feature_cols:
        for lag in lags:
            # Create lagged column name: feature_t-1, feature_t-2, ...
            lagged_col_name = f"{feature}_t-{lag}"
            df[lagged_col_name] = (
                df.groupby(group_cols)[feature]
                  .shift(lag)
            )

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

def feature_same_region(df, exporter_region_col='exporter_region', importer_region_col='importer_region'):
    df['same_region'] = np.where(df[exporter_region_col] == df[importer_region_col], 1, 0)
    return df

def feature_relative_population(df):
    df['population_exporter/importer'] = df['exporter_population'] / df['importer_population']
    return df

def feature_relative_gdp(df):
    df['gdp_exporter/importer'] = df['exporter_gdp_usd'] / df['importer_gdp_usd']
    return df

def feature_logistics_gap(df):
    df['logistics_index_gap'] = df['exporter_logistics_index'] - df['importer_logistics_index']
    return df

def feature_any_trade_agreement(df):
    df['any_trade_agreement'] = (
        df[['rta','fta','cu','psa','eia']].max(axis=1)
    )
    return df

def feature_political_events(df):
     # COVID period (2020–2021)
    df['covid_period'] = (df['year'].between(2020, 2021)).astype(int)

    # Russia–Ukraine war period (2022 onward)
    df['war_period'] = (df['year'] >= 2022).astype(int)
    df['russia_sanctions'] = (df['year'] >= 2014).astype(int)

    df['industrial_commodity_slowdown'] = df['year'].between(2014, 2016).astype(int)

    return df


def feature_top10_percent_trade(df):
    trade_threshold = df['trade_value_usd'].quantile(0.9)
    df['top10_percent_trade'] = np.where(df['trade_value_usd'] >= trade_threshold, 1, 0)
    return df

def feature_trade_volatility(df, group_cols=['exporter_name', 'importer_name', 'hs2'], time_col='year', window=3):
    df = df.sort_values(group_cols + [time_col])
    df['trade_value_usd_volatility'] = (
        df.groupby(group_cols)['trade_value_usd']
          .rolling(window=window)
          .std()
          .reset_index(level=group_cols, drop=True)
    )
    return df

def feature_log_transform(df, features):
    for feature in features:
        if feature in df.columns:
            # Handle negatives
            if (df[feature] < 0).any():
                print(f"Warning: Negative values found in {feature}. Clipping to 0 for log transform.")
                df[feature] = df[feature].clip(lower=0)
            df[f"{feature}_log"] = np.log1p(df[feature])
    return df


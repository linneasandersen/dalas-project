
from src.data.features import encode_hs2, engineer_lagged_features, panel_train_val_test_split
from src.data.main import PIPELINE_CONFIG, load_lagged_data, load_merged_data

from src.utils.io import save_df

from src.config import MERGED_DIR

from src.models.baseline import baseline_model_OLS


def model():
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



from src.data.features import encode_hs2, engineer_lagged_features
from src.models.split import panel_train_val_test_split, random_train_val_test_split
from src.data.loaders import load_lagged_data

from src.utils.io import save_df

from src.config import MERGED_DIR

from src.models.baseline import baseline_model_OLS

# ---------------------------------------------------

PIPELINE_CONFIG = {
    "engineer_lagged_features": True,
    "train_baseline": True,
}

# ---------------------------------------------------

def model(df):
    NUMBER_OF_LAGS = 4

    if PIPELINE_CONFIG["engineer_lagged_features"]:
        feature_cols = [
                        'exporter_temp_change_c',
                        'importer_temp_change_c', 
                        'trade_value_usd'
                        ]
        merged_df = engineer_lagged_features(
            df,
            group_cols=['exporter_id', 'importer_id', 'hs2'],
            time_col='year',
            feature_cols=feature_cols,
            lags=range(1, NUMBER_OF_LAGS + 1)
        )
        save_df(merged_df, f"oec_trade_lagged_test_{NUMBER_OF_LAGS}", MERGED_DIR / "lagged")
        print(merged_df.head())
        print(merged_df.columns)

    if PIPELINE_CONFIG["train_baseline"]:
    # Load merged lagged data
        merged_df = load_lagged_data(load_latest=True)

        # Step 1: One-hot encode hs2 
        primary_df_encoded = encode_hs2(merged_df, hs_col='hs2')

        # Step 2: Define features from the number of lags

        base_features = [
            'exporter_temp_change_c',
            'importer_temp_change_c',
        ]

        lagged_features = []

        for lag in range(1, NUMBER_OF_LAGS + 1):
            lagged_features.extend([
                f'exporter_temp_change_c_t-{lag}',
                f'importer_temp_change_c_t-{lag}',
                f'trade_value_usd_t-{lag}',
            ])

        lagged_features = lagged_features + [c for c in primary_df_encoded.columns if c.startswith('hs2_')]  # include hs2 dummies

        feature_cols = base_features + lagged_features

        # Step 3: Train/val/test split
        #train_df, val_df, test_df = panel_train_val_test_split(primary_df_encoded)
        train_df, val_df, test_df = random_train_val_test_split(primary_df_encoded)

        # Step 4: Train baseline OLS
        model, val_rmse = baseline_model_OLS(train_df, val_df, feature_cols, target_col='trade_value_usd', log_transform=True)



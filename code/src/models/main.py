
import time
from turtle import pd
from src.models.explainability import plot_feature_importance
from src.models.trees import train_random_forest, train_xgboost, tune_xgboost_random, tune_xgboost_temporal
from src.data.features import encode_hs2, engineer_lagged_features
from src.models.split import panel_train_val_test_split, random_train_val_test_split
from src.data.loaders import load_lagged_data
from src.models.error import analyze_errors

from src.utils.io import save_df

from src.config import LAGGED_DIR

from src.models.baseline import baseline_model_OLS

# ---------------------------------------------------

PIPELINE_CONFIG = {
    "engineer_lagged_features": False,
    "encode_features_split": True,
    "train_baseline": False,
    "train_rf": False,
    "train_xgb": False,
    "tune_xgb": False,
    "error_analysis": True,
    "feature_importance": True,
}

# ---------------------------------------------------

def model(df):
    NUMBER_OF_LAGS = 5
    TRAIN_DF_TEMPORAL = None
    VAL_DF_TEMPORAL = None
    TEST_DF_TEMPORAL = None
    TRAIN_DF_RANDOM = None
    VAL_DF_RANDOM = None
    TEST_DF_RANDOM = None
    FEATURE_COLS = None
    TARGET_COL = 'trade_value_usd'
    ONE_HOT_COL = 'product_category'
    MODEL_RANDOM = None
    MODEL_TEMPORAL = None


    if PIPELINE_CONFIG["engineer_lagged_features"]:
        feature_cols = [
                        'trade_value_usd',
                        'trade_value_usd_volatility',
                        'top10_percent_trade',
                        'pc1_importer',
                        'pc2_importer',
                        'pc1_exporter',
                        'pc2_exporter',
                        'exporter_temp_change_c',
                        'importer_temp_change_c',
                        'importer_temp_change_std_c',
                        'exporter_temp_change_std_c'
                        ]
        
        merged_df = engineer_lagged_features(
            df,
            group_cols=['exporter_id', 'importer_id', 'hs2'],
            time_col='year',
            feature_cols=feature_cols,
            lags=range(1, NUMBER_OF_LAGS + 1)
        )

        save_df(merged_df, f"data_lagged_{NUMBER_OF_LAGS}", LAGGED_DIR)
        print(merged_df.head())
        print(merged_df.columns)

    if PIPELINE_CONFIG["encode_features_split"]:
        t0 = time.time()
        # Load merged lagged data
        merged_df = load_lagged_data(NUMBER_OF_LAGS)
        # print columns without truncation
        # print(list(merged_df.columns))


        # Step 1: One-hot encode hs2 
        df_encoded = encode_hs2(merged_df, hs_col=ONE_HOT_COL, prefix=ONE_HOT_COL)

        # Step 2: Define features from the number of lags

        base_features = [
                        'same_region',
                        'countries_distance_km',
                        'any_trade_agreement',
                        'cueia',
                        'war_period',
                        'industrial_commodity_slowdown',
                        ]
    
        lagged_features = []

        for lag in range(1, NUMBER_OF_LAGS + 1):
            lagged_features.extend([
                f'trade_value_usd_t-{lag}',
                f'trade_value_usd_volatility_t-{lag}',
                f'top10_percent_trade_t-{lag}',
                f'pc1_importer_t-{lag}',
                f'pc2_importer_t-{lag}',
                f'pc1_exporter_t-{lag}',
                f'pc2_exporter_t-{lag}',
                f'exporter_temp_change_c_t-{lag}',
                f'importer_temp_change_c_t-{lag}',
                f'importer_temp_change_std_c_t-{lag}',
                f'exporter_temp_change_std_c_t-{lag}',
            ])

        lagged_features = lagged_features + [c for c in df_encoded.columns if c.startswith(ONE_HOT_COL + '_')]  # include cluster dummies

        FEATURE_COLS = base_features + lagged_features

        print(list(FEATURE_COLS))

        # remove rows with NaN in any of the feature columns
        df_encoded = df_encoded.dropna(subset=FEATURE_COLS)

        # Step 3: Train/val/test split
        TRAIN_DF_TEMPORAL, VAL_DF_TEMPORAL, TEST_DF_TEMPORAL = panel_train_val_test_split(df_encoded)
        TRAIN_DF_RANDOM, VAL_DF_RANDOM, TEST_DF_RANDOM = random_train_val_test_split(df_encoded)

        t1 = time.time()
        print(f"Data encoding and splitting time: {t1 - t0:.2f} seconds")


    if PIPELINE_CONFIG["train_baseline"]:
        time0 = time.time()
        print("Training Baseline OLS Models...")
        
        model, metrics_temporal = baseline_model_OLS(TRAIN_DF_TEMPORAL, VAL_DF_TEMPORAL, FEATURE_COLS, target_col='trade_value_usd', log_transform=True)
        model_rand, metrics_random = baseline_model_OLS(TRAIN_DF_RANDOM, VAL_DF_RANDOM, FEATURE_COLS, target_col='trade_value_usd', log_transform=True)

        time1 = time.time()
        print(f"Baseline OLS training time: {time1 - time0:.2f} seconds")

        print("Using encoding: ", ONE_HOT_COL)
        print("TEMPORAL SPLIT BASELINE MODEL")
        print("Temporal Split Validation Metrics:", metrics_temporal)

        print("RANDOM SPLIT BASELINE MODEL")
        print("Random Split Validation Metrics:", metrics_random)

    if PIPELINE_CONFIG["train_rf"]:
        print("Training Random Forest Models...")
        t0 = time.time()

        model_rf, val_pred_rf, metrics_rf = train_random_forest(TRAIN_DF_TEMPORAL, VAL_DF_TEMPORAL, FEATURE_COLS, TARGET_COL)
        model_rf_rand, val_pred_rf_rand, metrics_rf_rand = train_random_forest(TRAIN_DF_RANDOM, VAL_DF_RANDOM, FEATURE_COLS, TARGET_COL)
        t1 = time.time()
        print(f"Random Forest training time: {t1 - t0:.2f} seconds")

        print("Using encoding: ", ONE_HOT_COL)
        print("TEMPORAL SPLIT RF")
        print("Random Forest Validation Metrics (Temporal Split):", metrics_rf)
        print("RANDOM SPLIT RF")
        print("Random Forest Validation Metrics (Random split):", metrics_rf_rand)
    

    if PIPELINE_CONFIG["train_xgb"]:
        print("Training XGBoost Models...")
        t0 = time.time()

        model_xgb, val_pred_xgb, metrics_xgb = train_xgboost(TRAIN_DF_TEMPORAL, VAL_DF_TEMPORAL, FEATURE_COLS, TARGET_COL)
        model_xgb_rand, val_pred_xgb_rand, metrics_xgb_rand = train_xgboost(TRAIN_DF_RANDOM, VAL_DF_RANDOM, FEATURE_COLS, TARGET_COL)
        t1 = time.time()
        print(f"XGBoost training time: {t1 - t0:.2f} seconds")

        print("Using encoding: ", ONE_HOT_COL)
        print("TEMPORAL SPLIT XGBOOST")
        print("XGBoost Validation Metrics (Temporal Split):", metrics_xgb)
        print("RANDOM SPLIT XGBOOST")
        print("XGBoost Validation Metrics (Random split):", metrics_xgb_rand)

    if PIPELINE_CONFIG["tune_xgb"]:
        best_xgb_temporal, val_pred_temporal, metrics_temporal, best_params_temporal = tune_xgboost_temporal(TRAIN_DF_TEMPORAL, VAL_DF_TEMPORAL, FEATURE_COLS, TARGET_COL)
        best_xgb_random, val_pred_random, metrics_random, best_params_random = tune_xgboost_random(TRAIN_DF_RANDOM, VAL_DF_RANDOM, FEATURE_COLS, TARGET_COL)
        
        MODEL_RANDOM = best_xgb_random
        MODEL_TEMPORAL = best_xgb_temporal
        
        print("Using encoding: ", ONE_HOT_COL)
        print("TEMPORAL SPLIT TUNED XGBOOST")
        print("Tuned XGBoost Validation Metrics (Temporal Split):", metrics_temporal)
        print("Best Parameters (Temporal Split):", best_params_temporal)

        print("RANDOM SPLIT TUNED XGBOOST")
        print("Tuned XGBoost Validation Metrics (Random split):", metrics_random)
        print("Best Parameters (Random split):", best_params_random)

    if PIPELINE_CONFIG["error_analysis"]:
        OBJECTIVE = 'reg:tweedie'
        #OBJECTIVE = 'reg:squarederror'
        best_params_temporal = {'subsample': 1.0, 'n_estimators': 800, 'min_child_weight': 3, 'max_depth': 3, 'learning_rate': 0.05, 'colsample_bytree': 0.8}
        MODEL_TEMPORAL, _, metrics_temporal = train_xgboost(TRAIN_DF_TEMPORAL, VAL_DF_TEMPORAL, FEATURE_COLS, TARGET_COL, objective=OBJECTIVE, **best_params_temporal)
        MODEL_RANDOM, _, metrics_random = train_xgboost(TRAIN_DF_RANDOM, VAL_DF_RANDOM, FEATURE_COLS, TARGET_COL, objective=OBJECTIVE)

        print("Using encoding: ", ONE_HOT_COL)
        print("TEMPORAL SPLIT XGBOOST")
        print("XGBoost Validation Metrics (Temporal Split):", metrics_temporal)
        print("RANDOM SPLIT XGBOOST")
        print("XGBoost Validation Metrics (Random split):", metrics_random)

        analyze_errors(
            model_temporal=MODEL_TEMPORAL,
            model_random=MODEL_RANDOM,
            val_df_temporal=VAL_DF_TEMPORAL,
            test_df_temporal=TEST_DF_TEMPORAL,
            val_df_random=VAL_DF_RANDOM,
            test_df_random=TEST_DF_RANDOM,
            feature_cols=FEATURE_COLS,
            target_col=TARGET_COL,
            one_hot_col=ONE_HOT_COL,
            objective=OBJECTIVE
        )

        if PIPELINE_CONFIG["feature_importance"]:
            print("Computing feature importance...")

            feature_importance = plot_feature_importance(MODEL_TEMPORAL, FEATURE_COLS)
            print("Feature Importance:")
            print(feature_importance)
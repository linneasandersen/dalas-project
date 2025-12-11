import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ---------------------------------------------------
# Metrics
# ---------------------------------------------------

def compute_mape_smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    nonzero_mask = y_true != 0
    mape = np.mean(
        np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])
    ) * 100

    smape = np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))
    ) * 100

    return mape, smape


# ---------------------------------------------------
# Baseline model
# ---------------------------------------------------

def baseline_model_OLS(train_df, val_df, feature_cols, target_col='value', log_transform=True):
    lagged_features = [col for col in feature_cols if '_lag_' in col]

    print(f"Training before removing NaNs: {len(train_df)} rows")
    print(f"Validation before removing NaNs: {len(val_df)} rows")

    train_clean = train_df.dropna(subset=lagged_features).copy()
    val_clean = val_df.dropna(subset=lagged_features).copy()

    print(f"Training on {len(train_clean)} rows, validating on {len(val_clean)} rows")

    if log_transform:
        train_clean.loc[:, target_col + '_log'] = np.log1p(train_clean[target_col])
        val_clean.loc[:, target_col + '_log'] = np.log1p(val_clean[target_col])

        for f in lagged_features:
            train_clean.loc[:, f + '_log'] = np.log1p(train_clean[f].clip(lower=0))
            val_clean.loc[:, f + '_log'] = np.log1p(val_clean[f].clip(lower=0))

        feature_cols_used = [f + '_log' if f in lagged_features else f for f in feature_cols]
        target_col_used = target_col + '_log'
    else:
        feature_cols_used = feature_cols
        target_col_used = target_col

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])

    pipeline.fit(train_clean[feature_cols_used], train_clean[target_col_used])

    val_preds_log = pipeline.predict(val_clean[feature_cols_used])
    val_preds = np.expm1(val_preds_log) if log_transform else val_preds_log

    rmse = np.sqrt(mean_squared_error(val_clean[target_col], val_preds))
    rmse_log = np.sqrt(mean_squared_error(
        val_clean[target_col + "_log"], val_preds_log
    ))

    mape, smape = compute_mape_smape(val_clean[target_col], val_preds)

    print(f"MAPE: {mape:.2f}%")
    print(f"SMAPE: {smape:.2f}%")
    print(f"RMSE: {rmse}")
    print(f"RMSE (log): {rmse_log}")

    factor_error = np.expm1(rmse_log) + 1
    print(f"Avg factor error ~{factor_error:.2f}x")

    return pipeline, rmse

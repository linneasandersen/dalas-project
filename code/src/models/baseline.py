import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.models.metrics import compute_mape_smape, evaluate

# ---------------------------------------------------
# Baseline model
# ---------------------------------------------------

def baseline_model_OLS(train_df, val_df, feature_cols, target_col='trade_value_usd', log_transform=True):
    lagged_features = [col for col in feature_cols if '_t-' in col]

    train_clean = train_df.dropna(subset=lagged_features).copy()
    val_clean = val_df.dropna(subset=lagged_features).copy()

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

    MAX_LOG = 700  # float64 limit is ~709

    val_preds_log = pipeline.predict(val_clean[feature_cols_used])
    val_preds_log_clipped = np.clip(val_preds_log, None, MAX_LOG)
    val_preds = np.expm1(val_preds_log_clipped) if log_transform else val_preds_log_clipped

    metrics = evaluate(val_clean[target_col], val_preds)

    return pipeline, metrics

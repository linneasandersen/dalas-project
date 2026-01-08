import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from src.models.metrics import evaluate
from sklearn.model_selection import KFold

# define which features to log-transform
LOG_FEATURES = [
    "countries_distance_km",
    "trade_value_usd_t-1",
    "trade_value_usd_t-2",
    "trade_value_usd_t-3",
    "trade_value_usd_volatility_t-1",
    "trade_value_usd_volatility_t-2",
    "trade_value_usd_volatility_t-3",
]


def log_transform_features(df, feature_cols):
    """Return a copy of df with log1p applied to selected features."""
    df = df.copy()
    for col in feature_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))
    return df


def train_random_forest(train_df, val_df, feature_cols, target_col,
                        n_estimators=500, max_depth=None, min_samples_split=2,
                        random_state=42, n_jobs=-1):

    # log-transform target and features
    train_df = log_transform_features(train_df, LOG_FEATURES)
    val_df = log_transform_features(val_df, LOG_FEATURES)

    y_train = np.log1p(train_df[target_col].clip(lower=0))
    y_val = np.log1p(val_df[target_col].clip(lower=0))

    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=n_jobs
    )

    model.fit(X_train, y_train)

    val_pred_log = model.predict(X_val)
    # transform predictions back to original scale
    val_pred = np.expm1(val_pred_log)
    metrics = evaluate(val_df[target_col], val_pred)

    return model, val_pred, metrics


def train_xgboost(train_df, val_df, feature_cols, target_col, objective='reg:tweedie', subsample=0.9, n_estimators=500, min_child_weight=1, max_depth=6, learning_rate=0.05, colsample_bytree=0.9,
                  random_state=42, n_jobs=-1):

    train_df = train_df.dropna(subset=feature_cols + [target_col])
    val_df = val_df.dropna(subset=feature_cols + [target_col])

    # log-transform target and features
    train_df = log_transform_features(train_df, LOG_FEATURES)
    val_df = log_transform_features(val_df, LOG_FEATURES)

    if objective == 'reg:squarederror':
        y_train = np.log1p(train_df[target_col].clip(lower=0))
        y_val = np.log1p(val_df[target_col].clip(lower=0))
    else:
        y_train = train_df[target_col].clip(lower=0)
        y_val = val_df[target_col].clip(lower=0)

    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]

    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective=objective,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method="hist"
    )

    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    if objective == 'reg:squarederror':
        val_pred = np.expm1(val_pred)

    metrics = evaluate(val_df[target_col], val_pred)

    return model, val_pred, metrics


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from src.models.metrics import evaluate

# ---------------------------
# Random Forest tuning
# ---------------------------
def tune_random_forest(train_df, val_df, feature_cols, target_col,
                       n_iter=20, random_state=42):
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    rf = RandomForestRegressor(random_state=random_state, n_jobs=-1)

    param_dist = {
        'n_estimators': [300, 500, 700, 1000],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        'max_features': ['sqrt', 'log2', None]
    }

    rs = RandomizedSearchCV(
        rf, param_distributions=param_dist,
        n_iter=n_iter, cv=[(np.arange(len(X_train)), np.arange(len(X_val)))],
        scoring='neg_mean_squared_error', random_state=random_state,
        n_jobs=-1, verbose=1
    )

    # Fit on training, evaluate on validation
    rs.fit(X_train, y_train)
    best_rf = rs.best_estimator_
    val_pred = best_rf.predict(X_val)
    metrics = evaluate(y_val, val_pred)

    return best_rf, val_pred, metrics

# ---------------------------
# XGBoost tuning
# ---------------------------
def tune_xgboost_temporal(train_df, val_df, feature_cols, target_col,
                  n_iter=20, random_state=42):
    
    train_df = log_transform_features(train_df, LOG_FEATURES)
    val_df = log_transform_features(val_df, LOG_FEATURES)
    
    X_train = train_df[feature_cols]
    y_train = np.log1p(train_df[target_col])

    X_val = val_df[feature_cols]
    y_val = np.log1p(val_df[target_col])

    xgb = XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',
        random_state=random_state,
        n_jobs=-1,
        verbosity=0
    )

    param_dist = {
        'n_estimators': [300, 500, 800],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7, 9],
        'subsample': [0.7, 0.8, 0.9, 1],
        'colsample_bytree': [0.7, 0.8, 0.9, 1],
        'min_child_weight': [1, 3, 5]
    }

    rs = RandomizedSearchCV(
        xgb, param_distributions=param_dist,
        n_iter=n_iter, cv=[(np.arange(len(X_train)), np.arange(len(X_val)))],
        scoring='neg_mean_squared_error',
        random_state=random_state,
        verbose=1,
        n_jobs=-1
    )

    # Fit using early stopping on validation
    rs.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    best_xgb = rs.best_estimator_
    val_pred_log = best_xgb.predict(X_val)
    val_pred = np.expm1(val_pred_log)
    metrics = evaluate(val_df[target_col], val_pred)
    best_params = rs.best_params_

    return best_xgb, val_pred, metrics, best_params


def tune_xgboost_random(train_df, val_df, feature_cols, target_col,
                              n_iter=20, random_state=42):
    train_df = log_transform_features(train_df, LOG_FEATURES)
    val_df = log_transform_features(val_df, LOG_FEATURES)
    
    X_train = train_df[feature_cols]
    y_train = np.log1p(train_df[target_col])

    X_val = val_df[feature_cols]
    y_val = np.log1p(val_df[target_col])

    xgb = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0
    )

    param_dist = {
        "n_estimators": [300, 500, 800],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7, 9],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5]
    }

    # Proper CV for RANDOM split data
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

    rs = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_mean_squared_error",
        random_state=random_state,
        verbose=1,
        n_jobs=-1
    )

    rs.fit(X_train, y_train)

    best_xgb = rs.best_estimator_

    # Evaluate on your held-out validation set
    val_pred_log = best_xgb.predict(X_val)
    val_pred = np.expm1(val_pred_log)
    metrics = evaluate(val_df[target_col], val_pred)  
    best_params = rs.best_params_

    return best_xgb, val_pred, metrics, best_params

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from src.models.metrics import evaluate
from src.models.trees import log_transform_features, LOG_FEATURES

import matplotlib.pyplot as plt
import seaborn as sns

def analyze_war_impact(val_df, target_col, prediction_col='predicted_value', war_start_year=2022):
    """
    Compares model performance Pre-War vs. During-War for Conflict Zones.
    """
    # 1. Define Conflict Actors & Products
    conflict_exporters = ['Russia', 'Ukraine', 'Belarus'] # Russia, Ukraine, Belarus
    # Add major importers if you want (e.g., 'deu' for gas dependency)
    
    sensitive_products = ['product_category_Minerals', 'product_category_Agricultural', 'product_category_Metals']

    # 2. Create Masks
    is_conflict_zone = val_df['exporter_name'].isin(conflict_exporters)
    is_sensitive_prod = val_df['product_category'].isin(sensitive_products)
    is_war_period = val_df['year'] >= war_start_year
    
    # 3. Create Four Buckets
    # A: Conflict Zone (During War)
    # B: Conflict Zone (Pre-War)
    # C: Rest of World (During War)
    # D: Rest of World (Pre-War)
    
    # We focus on the "Conflict Shock" (A vs B)
    conflict_df = val_df[is_conflict_zone & is_sensitive_prod].copy()
    
    if conflict_df.empty:
        print("No conflict data found.")
        return

    print(f"=== WAR IMPACT ANALYSIS (Exporters: {conflict_exporters}) ===")
    
    # Calculate Metrics for Pre-War vs War
    results = []
    for period, data in conflict_df.groupby(is_war_period):
        period_name = "WAR PERIOD" if period else "PRE-WAR"
        
        print(f"\n--- {period_name} ({len(data)} rows) ---")
        
        metrics = evaluate(data[target_col], data[prediction_col])
        print(metrics)

        bias = (data[target_col] - data[prediction_col]).mean()
        avg_val = data[target_col].mean()
        pred_val = data[prediction_col].mean()
        
        results.append({
            'Period': period_name,
            'Count': len(data),
            'Avg Actual': avg_val,
            'Avg Predicted': pred_val,
            'Mean Bias ($)': bias, # Positive = Underprediction, Negative = Overprediction
            'Bias %': (bias / avg_val) * 100
        })
        
    # Output Table
    res_df = pd.DataFrame(results)
    pd.options.display.float_format = '{:,.0f}'.format
    print(res_df)
    
    return conflict_df

def analyze_error_by_quantile(df, target_col, prediction_col='prediction_value', quantile=0.90):
    """
    Splits the data into Bottom 90% (Small/Medium) and Top 10% (Giants)
    and reports metrics for each.
    """
    # 1. Determine the Threshold (e.g., the value separating the top 10%)
    threshold = df[target_col].quantile(quantile)
    
    print(f"\n=== QUANTILE ANALYSIS (Threshold: ${threshold:,.0f}) ===")
    
    # 2. Split Data
    df_small = df[df[target_col] <= threshold].copy() # Bottom 90%
    df_giant = df[df[target_col] > threshold].copy()  # Top 10%
    
    # 3. Helper to print metrics
    def print_bucket_metrics(name, subset):
        if subset.empty:
            print(f"{name}: No data.")
            return
        
        evaluated_metrics = evaluate(subset[target_col], subset[prediction_col])
        bias = (subset[target_col] - subset[prediction_col]).mean()
        
        
        print(f"\n--- {name} ({len(subset)} rows) ---")
        print(evaluated_metrics)
        print(f"Mean Bias: ${bias:,.0f}  <-- Positive = Underprediction")
        print(f"Avg Value: ${subset[target_col].mean():,.0f}")

    # 4. Run Analysis
    print_bucket_metrics(f"Bottom {quantile*100:.0f}% (Normal Trades)", df_small)
    print_bucket_metrics(f"Top {(1-quantile)*100:.0f}% (Giant Trades)", df_giant)

# --- HOW TO RUN IT ---
# Ensure your val_df has the 'prediction' column from your Tweedie model
# val_df_temporal['prediction'] = val_pred_tweedie


def plot_trade_route_history(df, exporter, importer, product, target_col, pred_col):
    """
    Plots the actual vs predicted trade values over time for a specific route.
    """
    # Filter for the specific route
    subset = df[
        (df['exporter_id'] == exporter) & 
        (df['importer_id'] == importer) & 
        (df['product_category'] == product)
    ].sort_values('year')
    
    if subset.empty:
        print(f"No data found for {exporter} -> {importer} ({product})")
        return

    plt.figure(figsize=(10, 5))
    
    # Plot Actual vs Predicted
    plt.plot(subset['year'], subset[target_col], marker='o', label='Actual Trade', linewidth=2)
    plt.plot(subset['year'], subset[pred_col], marker='x', linestyle='--', label='Predicted Trade', color='red', linewidth=2)
    
    # Formatting
    plt.title(f"Trade Route Analysis: {exporter.upper()} -> {importer.upper()} ({product})")
    plt.xlabel("Year")
    plt.ylabel("Trade Value (USD)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Use standard scaling to make large numbers readable
    plt.ticklabel_format(style='plain', axis='y') 
    
    plt.show()

def analyze_error_by_category(df, target_col, pred_col):
    """
    Calculates detailed error metrics grouped by product category.
    """
    results = []
    
    for category, group in df.groupby('product_category'):
        y_true = group[target_col]
        y_pred = group[pred_col]
        
        # Calculate metrics
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Mean Bias (Are we generally over or under predicting?)
        # Positive = Underpredicting (True > Pred)
        # Negative = Overpredicting (Pred > True)
        bias = np.mean(y_true - y_pred) 
        
        results.append({
            'Category': category,
            'Count': len(group),
            'RMSE': rmse,
            'MAPE (%)': mape,
            'Mean Bias ($)': bias
        })
    
    # Create a clean dataframe
    metrics_df = pd.DataFrame(results).sort_values('RMSE', ascending=False)
    
    # Display formatted
    pd.set_option('display.float_format', '{:,.0f}'.format)
    print("\n=== Error Analysis by Product Category ===")
    print(metrics_df)
    
    return metrics_df


def analyze_errors(model_temporal, model_random,
                   val_df_temporal: pd.DataFrame, test_df_temporal: pd.DataFrame,
                   val_df_random: pd.DataFrame, test_df_random: pd.DataFrame,
                   feature_cols: list, target_col: str, one_hot_col: str, 
                   objective='reg:tweedie'): # Default to your new winner
    
    # -------------------------------
    # 0. SAFETY: Create local copies
    # -------------------------------
    # We copy so we don't accidentally mutate the global dataframes
    val_df_temp_proc = val_df_temporal.copy()
    test_df_temp_proc = test_df_temporal.copy()
    val_df_rand_proc = val_df_random.copy()
    test_df_rand_proc = test_df_random.copy()

    # -------------------------------
    # 1. Transform Features (Keep log transform for features!)
    # -------------------------------
    val_df_temp_proc = log_transform_features(val_df_temp_proc, LOG_FEATURES)
    test_df_temp_proc = log_transform_features(test_df_temp_proc, LOG_FEATURES)
    val_df_rand_proc = log_transform_features(val_df_rand_proc, LOG_FEATURES)
    test_df_rand_proc = log_transform_features(test_df_rand_proc, LOG_FEATURES)

    # -------------------------------
    # 2. Collect predictions
    # -------------------------------
    # Helper to handle the logic cleanly
    def get_pred(model, df, objective):
        raw_output = model.predict(df[feature_cols])
        if objective == 'reg:squarederror':
            return np.expm1(raw_output) # Inverse log
        else:
            return raw_output # Tweedie outputs raw scale directly

    temporal_val_pred = get_pred(model_temporal, val_df_temp_proc, objective)
    temporal_test_pred = get_pred(model_temporal, test_df_temp_proc, objective)
    random_val_pred = get_pred(model_random, val_df_rand_proc, objective)
    random_test_pred = get_pred(model_random, test_df_rand_proc, objective)

    # -------------------------------
    # 3. Decode one-hot (Use the PROCESSED dataframes)
    # -------------------------------
    encoded_cols = [c for c in val_df_temp_proc.columns if c.startswith(one_hot_col + '_')]
    val_df_temp_proc[one_hot_col] = val_df_temp_proc[encoded_cols].idxmax(axis=1)
    val_df_rand_proc[one_hot_col] = val_df_rand_proc[encoded_cols].idxmax(axis=1)

    # -------------------------------
    # 4. Evaluate metrics (Compare Prediction vs. ORIGINAL Target)
    # -------------------------------
    # Note: We use the original 'target_col' which should be in the df
    metrics_temporal_val = evaluate(val_df_temp_proc[target_col], temporal_val_pred)
    metrics_temporal_test = evaluate(test_df_temp_proc[target_col], temporal_test_pred)

    metrics_random_val = evaluate(val_df_rand_proc[target_col], random_val_pred)
    metrics_random_test = evaluate(test_df_rand_proc[target_col], random_test_pred)

    print("=== Temporal Split ===")
    print("Validation Metrics:", metrics_temporal_val)
    print("Test Metrics:", metrics_temporal_test)

    print("\n=== Random Split ===")
    print("Validation Metrics:", metrics_random_val)
    print("Test Metrics:", metrics_random_test)

    # -------------------------------
    # 3. Compute residuals
    # -------------------------------
    val_df_temp_proc['residuals'] = val_df_temp_proc[target_col] - temporal_val_pred
    test_df_temp_proc['residuals'] = test_df_temp_proc[target_col] - temporal_test_pred
    val_df_rand_proc['residuals'] = val_df_rand_proc[target_col] - random_val_pred
    test_df_rand_proc['residuals'] = test_df_rand_proc[target_col] - random_test_pred

    val_df_temp_proc['abs_error'] = val_df_temp_proc['residuals'].abs()
    val_df_rand_proc['abs_error'] = val_df_rand_proc['residuals'].abs()

    # -------------------------------
    # 4. Top 10 largest errors
    # -------------------------------
    print("\nTop 10 Temporal Validation Errors:")
    print(val_df_temp_proc.nlargest(10, 'abs_error')[['exporter_id', 'importer_id', one_hot_col, target_col, 'residuals']])

    print("\nTop 10 Random Validation Errors:")
    print(val_df_rand_proc.nlargest(10, 'abs_error')[['exporter_id', 'importer_id', one_hot_col, target_col, 'residuals']])

    # -------------------------------
    # 5. Correlation of residuals with features
    # -------------------------------
    residual_corr_temporal = val_df_temp_proc[feature_cols + ['residuals']].corr()['residuals'].sort_values(ascending=False)
    residual_corr_random = val_df_rand_proc[feature_cols + ['residuals']].corr()['residuals'].sort_values(ascending=False)

    print("\nTemporal Residuals Correlation with Features:\n", residual_corr_temporal.head(10))
    print("\nRandom Residuals Correlation with Features:\n", residual_corr_random.head(10))

    # -------------------------------
    # 6. Scatter plot predicted vs true
    # -------------------------------
    plt.figure(figsize=(8,8))
    sns.scatterplot(x=val_df_temp_proc[target_col], y=temporal_val_pred, alpha=0.3)
    plt.plot([0, val_df_temp_proc[target_col].max()], [0, val_df_temp_proc[target_col].max()], 'r--')
    plt.xlabel("True Trade Value")
    plt.ylabel("Predicted Trade Value")
    plt.title("Temporal Split: True vs Predicted")
    plt.show()

    plt.figure(figsize=(8,8))
    sns.scatterplot(x=val_df_rand_proc[target_col], y=random_val_pred, alpha=0.3)
    plt.plot([0, val_df_rand_proc[target_col].max()], [0, val_df_rand_proc[target_col].max()], 'r--')
    plt.xlabel("True Trade Value")
    plt.ylabel("Predicted Trade Value")
    plt.title("Random Split: True vs Predicted")
    plt.show()

    # Plot residuals vs predicted values
    plt.scatter(temporal_val_pred, val_df_temp_proc['residuals'], alpha=0.3)
    plt.xscale('log')
    plt.yscale('symlog') # Allows negative values on log scale
    plt.xlabel("Predicted Value (Log Scale)")
    plt.ylabel("Residuals (Log Scale)")
    plt.title(f"Temporal Split: Residuals vs Predicted ({objective})")


    mean_bias = val_df_temp_proc['residuals'].mean()
    print(f"Mean Bias: {mean_bias:,.0f}")

    # Filter for Canada to USA Minerals
    # subset = val_df_temp_proc[
    #     (val_df_temp_proc['exporter_id'] == 'can') & 
    #     (val_df_temp_proc['importer_id'] == 'usa') & 
    #     (val_df_temp_proc['product_category'] == 'product_category_Minerals')
    # ].sort_values('year')

    val_df_temp_proc['predicted_value'] = temporal_val_pred 
    # plt.plot(subset['year'], subset[target_col], label='Actual')
    # plt.plot(subset['year'], subset['predicted_value'], label='Predicted')
 

    # 1. Check the massive Canada -> USA Minerals error
    #plot_trade_route_history(val_df_temp_proc, 'can', 'usa', 'product_category_Minerals', target_col, 'predicted_value')

    # 2. Check the China -> USA Manufactured error
    #plot_trade_route_history(val_df_temp_proc, 'chn', 'usa', 'product_category_Manufactured', target_col, 'predicted_value')

    cat_metrics_temp = analyze_error_by_category(val_df_temp_proc, target_col, 'predicted_value')
    print("\nCategory-wise Error Metrics:\n", cat_metrics_temp)

    analyze_error_by_quantile(val_df_temp_proc, 'trade_value_usd', 'predicted_value', quantile=0.90)

    conflict_data = analyze_war_impact(val_df_temp_proc, 'trade_value_usd')
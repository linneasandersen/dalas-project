import scipy.stats as stats
import numpy as np

def test_categorical_relevance(df, categorical_col, target_col='trade_value_usd'):
    """
    Performs One-Way ANOVA to see if a categorical column impacts the target.
    """

    # log transform the target column to reduce skewness
    target_col_log = f"{target_col}_log"
    df[target_col_log] = np.log1p(df[target_col])

    # 1. Drop NaNs so the test doesn't break
    data = df.dropna(subset=[categorical_col, target_col])
    
    # 2. Group the target values by category
    groups = [data[target_col_log][data[categorical_col] == category] for category in data[categorical_col].unique()]
    
    # 3. Run ANOVA
    f_stat, p_value = stats.f_oneway(*groups)
    
    print(f"--- ANOVA Results for: {target_col} vs {categorical_col} ---")
    print(f"F-Statistic: {f_stat:.2f}")
    print(f"P-Value: {p_value:.4e}")
    
    if p_value < 0.05:
        print(">> RESULT: SIGNIFICANT. This category strongly affects Trade Value.\n")
    else:
        print(">> RESULT: NOT Significant. You might be able to drop this column.\n")

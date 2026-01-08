import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_feature_importance(model, feature_cols, top_n=20, figsize=(10, 8)):
    """
    Plots the XGBoost Feature Importance (Gain).
    """
    # 1. Get the booster from the sklearn wrapper
    booster = model.get_booster()
    
    # 2. Get importance (explicitly asking for 'gain')
    # This returns a dict like {'f0': 123.4, 'f1': 56.7}
    importance_map = booster.get_score(importance_type='gain')
    
    # 3. Map feature names (XGBoost sometimes uses f0, f1...)
    # We map them back to your actual column names
    feature_map = {f'f{i}': col for i, col in enumerate(feature_cols)}
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': [feature_map.get(k, k) for k in importance_map.keys()],
        'Gain': list(importance_map.values())
    })
    
    # Sort and normalize
    importance_df = importance_df.sort_values(by='Gain', ascending=False).head(top_n)
    
    # 4. Plot
    plt.figure(figsize=figsize)
    sns.barplot(data=importance_df, x='Gain', y='Feature', palette='viridis')
    plt.title('XGBoost Feature Importance (Type: Gain)', fontsize=14)
    plt.xlabel('Average Gain (Contribution to Error Reduction)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return importance_df

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_heatmap(df: pd.DataFrame, features: list, title: str = "Correlation Heatmap", log=False) -> None:
    """
    Plot a correlation heatmap for the specified features in the dataframe.
    
    Parameters:
        df: DataFrame containing the data.
        features: List of feature column names to include in the correlation matrix.
        title: Title of the heatmap.
        log: Boolean, if True, applies log1p transform to skewed features.
    """
    print("Calculating correlation matrix...")

    # Create a copy so we don't modify the original dataframe outside
    df_plot = df.copy()
    
    # Start with the original list of features we want to plot
    features_to_plot = features.copy()

    if log:
        # Added distance to this list as it's critical for trade models
        features_to_transform = [
            'trade_value_usd',
            'exporter_gdp_usd',
            'importer_gdp_usd',
            'exporter_gdp_per_capita_usd',
            'importer_gdp_per_capita_usd',
            'gdp_exporter/importer',
            'exporter_population',
            'importer_population',
            'population_exporter/importer',
            'countries_distance_km',
            'trade_value_usd_volatility',
            'exporter_land_area_km2',
            'importer_land_area_km2',
        ]

        # Create a mapping dictionary: { 'old_name': 'new_name' }
        name_mapping = {}

        for feature in features_to_transform:
            # Only process if the feature is actually in our DataFrame
            if feature in df_plot.columns:
                
                # Handle negatives
                if (df_plot[feature] < 0).any():
                    print(f"Warning: Negative values found in {feature}. Clipping to 0 for log transform.")
                    df_plot[feature] = df_plot[feature].clip(lower=0)
                
                # Apply log1p
                df_plot[feature] = np.log1p(df_plot[feature])
                
                # Rename the column in the DataFrame
                new_feature_name = f"{feature}_log"
                df_plot.rename(columns={feature: new_feature_name}, inplace=True)
                
                # Store the change in our mapping dictionary
                name_mapping[feature] = new_feature_name

        # Update the list of features to plot using the mapping
        # If a feature is in the map, use the new name; otherwise keep the old name
        features_to_plot = [name_mapping.get(x, x) for x in features]

    # Calculate correlation with the correct list of columns
    corr = df_plot[features_to_plot].corr()    

    # Plotting
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0,
                cbar_kws={"shrink": .8}, square=True, linewidths=.5)
    plt.title(title, fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    #print("Correlation matrix:\n", corr)
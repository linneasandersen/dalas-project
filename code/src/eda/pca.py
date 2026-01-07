import pandas as pd
from src.config import PRODUCT_NAMES_SHORT, PRODUCT_PALETTE
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def plot_pca_biplot(df, features, color_col=None, title="PCA"):
    """
    Performs PCA and plots a Biplot (Scores + Loadings).
    
    Parameters:
    - df: The dataframe.
    - features: List of numerical column names to use for PCA.
    - color_col: (Optional) Categorical column to color the points by (e.g., 'exporter_region').
    """
    print("Preparing data for PCA...")

    # 1. Subset and Clean Data
    # We combine features + color_col to drop NaNs together
    cols_to_use = features + ([color_col] if color_col else [])
    df_pca = df[cols_to_use].dropna().copy()
    
    # 2. Standardize the Features (Crucial for PCA)
    # PCA is sensitive to scale (e.g., GDP vs. Logistics Index), so we must scale to mean=0, std=1
    x = df_pca[features].values
    x = StandardScaler().fit_transform(x)
    
    # 3. Run PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    
    # Create a DataFrame for the Principal Components
    pc_df = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
    
    if color_col:
        pc_df[color_col] = df_pca[color_col].values

    # 4. Calculate Loadings (The Arrows)
    # This matrix tells us how much each original feature contributes to PC1 and PC2
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    # print loadings
    print("PCA Loadings:")
    for i, feature in enumerate(features):
        print(f"{feature}: PC1={loadings[i, 0]:.4f}, PC2={loadings[i, 1]:.4f}")
    
    # --- PLOTTING ---
    plt.figure(figsize=(14, 10))
    
    # A. Scatter Plot of the Scores (The Points)
    if color_col:
        sns.scatterplot(x='PC1', y='PC2', hue=color_col, data=pc_df, 
                        alpha=0.6, palette='viridis', s=60)
    else:
        plt.scatter(pc_df['PC1'], pc_df['PC2'], alpha=0.5, s=60)

    # B. Plot the Arrows (The Loadings)
    # We scale arrows to match the plot dimensions so they are visible
    max_pc = max(pc_df['PC1'].max(), pc_df['PC2'].max())
    scale_factor = max_pc * 0.8  # Scale arrows to 80% of the max axis value
    
    for i, feature in enumerate(features):
        # Coordinates for the arrow tip
        x_arrow = loadings[i, 0] * 3  # Multiplier usually needed to make arrows visible on large plots
        y_arrow = loadings[i, 1] * 3
        
        plt.arrow(0, 0, x_arrow, y_arrow, color='r', alpha=0.9, head_width=0.2)
        
        # Add text label for the arrow
        plt.text(x_arrow * 1.15, y_arrow * 1.15, feature, color='darkred', 
                 ha='center', va='center', fontsize=12, weight='bold')

    # Formatting
    explained_var = pca.explained_variance_ratio_
    print(f"Explained Variance: {explained_var}")
    plt.xlabel(f'Principal Component 1 ({explained_var[0]:.1%} Variance)', fontsize=14)
    plt.ylabel(f'Principal Component 2 ({explained_var[1]:.1%} Variance)', fontsize=14)
    plt.title(title, fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    if color_col:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def plot_hs2_pca_colored(df):
    print("Preparing Historical Product Profiles...")

    # --- 1. HISTORICAL DATA GENERATION (The 'Lag' Logic) ---
    # Group by year and hs2 to get stats for every year/product combo
    yearly_stats = df.groupby(['year', 'hs2']).agg({
        'trade_value_usd_log': ['mean', 'std'],
        'countries_distance_km': 'mean'
    })
    yearly_stats.columns = ['avg_value', 'volatility', 'avg_dist']
    yearly_stats = yearly_stats.reset_index()

    # Shift years: 2015 data becomes labeled '2016' so it acts as history
    yearly_stats['hist_year'] = yearly_stats['year'] + 1
    
    historical_lookup = yearly_stats[[
        'hist_year', 'hs2', 'avg_value', 'volatility', 'avg_dist'
    ]].rename(columns={
        'hist_year': 'year',
        'avg_value': 'hist_prod_value',
        'volatility': 'hist_prod_volatility',
        'avg_dist': 'hist_prod_distance'
    })

    # Merge history back to main dataframe
    df_with_hist = df.merge(historical_lookup, on=['year', 'hs2'], how='left')
    df_with_hist[['hist_prod_value', 'hist_prod_volatility', 'hist_prod_distance']] = \
        df_with_hist[['hist_prod_value', 'hist_prod_volatility', 'hist_prod_distance']].fillna(0)

    # --- 2. PREPARE PRODUCT SPACE FOR PLOTTING ---
    # For the plot, we want ONE row per product, using their overall historical profile
    product_stats = df_with_hist.groupby('hs2').agg({
        'hist_prod_value': 'mean',
        'hist_prod_volatility': 'mean',
        'hist_prod_distance': 'mean',
        'product_category': 'first'
    })
    # make hs2 a column (easier for seaborn hue mapping)
    product_stats = product_stats.reset_index()
    
    hist_features = ['hist_prod_value', 'hist_prod_volatility', 'hist_prod_distance']

    # --- 3. PCA ON HISTORICAL FEATURES ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(product_stats[hist_features])

    pca_prod = PCA(n_components=2)
    prod_components = pca_prod.fit_transform(X_scaled)
    
    product_stats['PC1'] = prod_components[:, 0]
    product_stats['PC2'] = prod_components[:, 1]

    # --- 4. VISUALIZATION ---
    plt.figure(figsize=(14, 10))
    # Build a palette using PRODUCT_PALETTE primary colors keyed by hs2 code
    palette_dict = {row['hs2']: PRODUCT_PALETTE.get(row['hs2'], {}).get('primary', '#444444')
                    for _, row in product_stats.iterrows()}
    sns.scatterplot(
        x='PC1', y='PC2', hue='hs2', 
        data=product_stats, s=120, alpha=0.95, palette=palette_dict, edgecolor='black'
    )
    # draw axis lines (x=0, y=0) so axes are visible
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)

    # Replace numeric HS2 labels with short product names in the legend
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    # Map numeric labels (strings like '10') to PRODUCT_NAMES_SHORT; leave others untouched
    mapped_labels = []
    for lab in labels:
        try:
            code = int(lab)
            mapped_labels.append(PRODUCT_NAMES_SHORT.get(code, lab))
        except Exception:
            mapped_labels.append(lab)
    # Rebuild legend with mapped labels and place outside plot
    if handles:
        ax.legend(handles=handles, labels=mapped_labels, title='Product', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add Vectors (Arrows)
    loadings = pca_prod.components_.T * np.sqrt(pca_prod.explained_variance_)
    # Print loadings and explained variance for inspection
    print("PCA Loadings (feature -> PC1, PC2):")
    for i, feature in enumerate(hist_features):
        print(f"{feature}: PC1={loadings[i, 0]:.4f}, PC2={loadings[i, 1]:.4f}")
    print(f"Explained variance ratio: {pca_prod.explained_variance_ratio_}")
    scale_factor = np.max(np.abs(prod_components)) * 0.8
    for i, feature in enumerate(hist_features):
        plt.arrow(0, 0, loadings[i, 0]*scale_factor, loadings[i, 1]*scale_factor, 
                  color='red', alpha=0.9, head_width=0.2)
        plt.text(loadings[i, 0]*scale_factor*1.2, loadings[i, 1]*scale_factor*1.2, 
                 feature, color='darkred', weight='bold')

    # Label products using names from your dictionary
    # Label products using names from your dictionary (offset slightly for readability)
    for i in range(product_stats.shape[0]):
        hs2_code = product_stats['hs2'].iloc[i]
        plt.text(product_stats['PC1'].iloc[i], product_stats['PC2'].iloc[i] + 0.1,
                 PRODUCT_NAMES_SHORT.get(hs2_code, str(hs2_code)), fontsize=8, alpha=0.7)

    plt.title("Historical Product Space", fontsize=16)
    plt.xlabel(f"PC1 ({pca_prod.explained_variance_ratio_[0]:.1%} Variance)")
    plt.ylabel(f"PC2 ({pca_prod.explained_variance_ratio_[1]:.1%} Variance)")
    plt.show()

    # --- 5. CLUSTERING ---
    kmeans = KMeans(n_clusters=4, random_state=0)
    product_stats['cluster'] = kmeans.fit_predict(product_stats[['PC1', 'PC2']])

    plt.figure(figsize=(14, 10))
    # Map clusters to a discrete palette and categories to their PRODUCT_PALETTE primary colors
    n_clusters = product_stats['cluster'].nunique()
    cluster_palette = sns.color_palette('Set2', n_clusters)
    cluster_color_map = {c: cluster_palette[i] for i, c in enumerate(sorted(product_stats['cluster'].unique()))}

    # category -> color (primary) mapping
    category_color_map = {}
    for cat in product_stats['product_category'].unique():
        sample = product_stats[product_stats['product_category'] == cat]
        if not sample.empty:
            sample_hs2 = sample['hs2'].iloc[0]
            category_color_map[cat] = PRODUCT_PALETTE.get(sample_hs2, {}).get('primary', '#444444')
        else:
            category_color_map[cat] = '#444444'

    # Force Metals category to use black for visibility if present
    if 'Metals' in category_color_map:
        category_color_map['Metals'] = '#000000'

    # Plot points using matplotlib so we can set facecolor (cluster) and edgecolor (category)
    for _, row in product_stats.iterrows():
        pc1, pc2 = row['PC1'], row['PC2']
        cluster = row['cluster']
        cat = row['product_category']
        face = cluster_color_map[cluster]
        edge = category_color_map.get(cat, '#444444')
        plt.scatter(pc1, pc2, s=220, facecolors=face, edgecolors=edge, linewidths=1.6, alpha=0.95)

    # ensure axes lines are visible on cluster plot too
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)

    # Add short product name labels for each point in the cluster plot
    for i in range(product_stats.shape[0]):
        hs2_code = product_stats['hs2'].iloc[i]
        plt.text(
            product_stats['PC1'].iloc[i],
            product_stats['PC2'].iloc[i] + 0.12,
            PRODUCT_NAMES_SHORT.get(hs2_code, str(hs2_code)),
            fontsize=8, alpha=0.95, ha='center'
        )

    # Build and place legends: one for clusters (face colors) and one for categories (edge colors)
    ax = plt.gca()
    cluster_handles = [mpatches.Patch(color=cluster_color_map[c], label=f'Cluster {c+1}') for c in sorted(cluster_color_map.keys())]
    category_handles = [mpatches.Patch(facecolor='white', edgecolor=color, label=cat, linewidth=2) for cat, color in category_color_map.items()]

    leg1 = ax.legend(handles=cluster_handles, title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.add_artist(leg1)
    ax.legend(handles=category_handles, title='Category', bbox_to_anchor=(1.02, 0.4), loc='upper left')

    plt.title("Product Space Clusters (face=color=cluster, edge=color=category)")
    plt.xlabel(f"PC1 ({pca_prod.explained_variance_ratio_[0]:.1%} Variance)")
    plt.ylabel(f"PC2 ({pca_prod.explained_variance_ratio_[1]:.1%} Variance)")
    plt.tight_layout()
    plt.show()

    # print cluster assignments
    print("Product Cluster Assignments:")
    for _, row in product_stats.iterrows():
        print(f"HS2 Code {row['hs2']} ({PRODUCT_NAMES_SHORT.get(row['hs2'], 'Unknown')}): Cluster {row['cluster']}")    
    
    df_with_cluster = df_with_hist.merge(
        product_stats[['hs2', 'cluster']], on='hs2', how='left'
    )

    # add also PC1 and PC2 to the main dataframe for potential use
    df_with_cluster = df_with_cluster.merge(
        product_stats[['hs2', 'PC1', 'PC2']], on='hs2', how='left'
    )

    return df_with_cluster # Return this for your prediction model




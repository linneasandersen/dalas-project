
from src.utils.io import save_df
from src.data.features import feature_log_transform
from src.data.loaders import load_latest_eda_data
from src.config import AGRICULTURAL_HS2_CODES, GOOGLE_DRIVE, MINERALS_HS2_CODES, MANUFACTURED_HS2_CODES, METALS_HS2_CODES, CHEMICALS_HS2_CODES
from src.eda.sanity_check import check_sanity
from src.eda.trade_value import countries_centered_plot, cumulative_share_plot, cumulative_share_plot_percent, histogram_trade_value_by_product_log_x, histogram_trade_values, histogram_trade_values_log_x, histogram_trade_value_by_product, trade_over_time, trade_over_time_products
from src.config import HS2_CODES
from src.eda.correlations import plot_correlation_heatmap
from src.eda.anova import test_categorical_relevance
from src.eda.pca import plot_hs2_pca_colored, plot_pca_biplot
from src.eda.boxplots import boxplot_trade_value_by_product, boxplot_trade_value_all

PIPELINE_CONFIG = {
    "sanity_check": False,
    "trade_value_histograms": False,
    "trade_value": False,
    "boxplots": False,
    "anova": False,
    "pca_biplot": False,
    "correlation_matrix": False,
    "return_df": True,
}

# ---------------------------------------------------
# 3. Main orchestration
# ---------------------------------------------------

def explore(df):
    processed_dir = GOOGLE_DRIVE / "processed" / "eda"

    if PIPELINE_CONFIG["sanity_check"]:
        check_sanity(df)
        
    if PIPELINE_CONFIG["trade_value_histograms"]:
        histogram_trade_values(df)
        histogram_trade_values_log_x(df)
        histogram_trade_value_by_product(df, hs2_code=15, color='sandybrown')
        histogram_trade_value_by_product_log_x(df, hs2_code=15, color='sienna')

    if PIPELINE_CONFIG["trade_value"]:
        countries_centered_plot(df, 0)
        countries_centered_plot(df, country_column='importer_name', country_index=0, color='lightcoral')
        cumulative_share_plot_percent(df)
        trade_over_time_products(df, products=HS2_CODES)
        trade_over_time_products(df, products=AGRICULTURAL_HS2_CODES, include_top_bottom=True, title="Agricultural Products Trade Over Time")
        trade_over_time_products(df, products=MINERALS_HS2_CODES, include_top_bottom=True, title="Mineral Products Trade Over Time")
        trade_over_time_products(df, products=MANUFACTURED_HS2_CODES[:3], include_top_bottom=True, title="Manufactured Products Trade Over Time")
        trade_over_time_products(df, products=METALS_HS2_CODES[:2] + METALS_HS2_CODES[3:4], include_top_bottom=True, title="Metal Products Trade Over Time")
        trade_over_time_products(df, products=CHEMICALS_HS2_CODES, include_top_bottom=True, title="Chemical Products Trade Over Time")

    if PIPELINE_CONFIG["boxplots"]:
        #boxplot_trade_value_by_product(df, log=False)
        #boxplot_trade_value_by_product(df, log=True)
        boxplot_trade_value_all(df, log=False)
        boxplot_trade_value_all(df, log=True)
    
    if PIPELINE_CONFIG["anova"]:
        test_categorical_relevance(df, 'exporter_name')
        test_categorical_relevance(df, 'importer_name')
        test_categorical_relevance(df, 'exporter_region')
        test_categorical_relevance(df, 'importer_region')
        test_categorical_relevance(df, 'same_region')
        test_categorical_relevance(df, 'hs2')
        test_categorical_relevance(df, 'product_category')
        test_categorical_relevance(df, 'year')
        test_categorical_relevance(df, 'any_trade_agreement')
        test_categorical_relevance(df, 'fta')
        test_categorical_relevance(df, 'cu')
        test_categorical_relevance(df, 'psa')
        test_categorical_relevance(df, 'eia')
        test_categorical_relevance(df, 'cueia')
        test_categorical_relevance(df, 'ftaeia')
        test_categorical_relevance(df, 'psaeia')
        test_categorical_relevance(df, 'covid_period')
        test_categorical_relevance(df, 'war_period')
        test_categorical_relevance(df, 'russia_sanctions')
        test_categorical_relevance(df, 'industrial_commodity_slowdown')

        import statsmodels.api as sm
        from statsmodels.formula.api import ols

        # two-way anova
        model = ols('trade_value_usd_log ~ C(exporter_region) * C(product_category)', data=df).fit()
        interaction_anova = sm.stats.anova_lm(model, typ=2)

        print(interaction_anova)

    if PIPELINE_CONFIG["pca_biplot"]:
        features = [
            'trade_value_usd',
            'exporter_gdp_usd',
            'exporter_population',
            'exporter_logistics_index',
            'exporter_gdp_per_capita_usd',
            'trade_value_usd_volatility',
            'exporter_land_area_km2'
        ]

        df = feature_log_transform(df, features)

        pca_features = [
            'exporter_gdp_usd_log', 
            'exporter_population_log', 
            'exporter_logistics_index', 
            'exporter_gdp_per_capita_usd_log',
            'exporter_land_area_km2_log'
        ]

        # Run the function
        df_country_pca = plot_pca_biplot(df, features=pca_features, color_col='exporter_region')
        print(df_country_pca[['PC1', 'PC2', 'exporter_name', 'year']].head(100))

        cluster_mapping = plot_hs2_pca_colored(df[df['year'].between(2010, 2020)])

        # Make sure df_country_pca has one row per exporter/year
        df_country_pca_unique = df_country_pca.groupby(['exporter_name','year']).agg({
            'PC1':'mean',
            'PC2':'mean'    
        }).reset_index()

        df_country_pca_unique_import = df_country_pca_unique.rename(columns={
            'exporter_name':'importer_name',
        })

        # Merge safely
        df_save = df.merge(
            df_country_pca_unique,
            on=['exporter_name','year'],
            how='left'
        ).rename(columns={'PC1':'pc1_exporter','PC2':'pc2_exporter'})

        print(df_save.columns)
        print(df_country_pca_unique.columns)

        df_save = df_save.merge(
            df_country_pca_unique_import,
            on=['importer_name','year'],
            how='left'
        ).rename(columns={'PC1':'pc1_importer','PC2':'pc2_importer'})

        print(df_save[['pc1_importer', 'pc2_importer', 'importer_name', 'year']].head(10))
        print(df_save[['pc1_exporter', 'pc2_exporter', 'exporter_name', 'year']].head(10))
        print(df_save[['pc1_exporter', 'pc2_exporter', 'exporter_name', 'pc1_importer', 'pc2_importer', 'importer_name', 'year']].head(10))

        # merge with cluster mapping
        df_save = df_save.merge(
            cluster_mapping,
            on='hs2',
            how='left'
        )

        df_save = df_save[[col for col in df_save.columns if not col.endswith('_log')]]

        # print example for year = 2023
        print(df_save[df_save['year'] == 2023][['hs2', 'cluster']].head(20))

        save_df(df_save, "data_pca_clusters", processed_dir)
    
    if PIPELINE_CONFIG["correlation_matrix"]:
        # load latest file from processed/eda 
        df_eda = load_latest_eda_data()

        gravity_features = [
            'trade_value_usd',            # Target
            'exporter_gdp_usd',           # Economic Mass
            'importer_gdp_usd',
            'exporter_population',        # Human Mass
            'importer_population',
            'exporter_land_area_km2',     # Physical Size
            'importer_land_area_km2',
            'countries_distance_km',      # Distance constraint
            'same_region',                # Geographic grouping
            'top10_percent_trade'         # Is this a "Major" trade partner?
        ]
        plot_correlation_heatmap(df_eda, gravity_features, title="Correlation Matrix of Trade Features", log=True)

        efficiency_features = [
            'trade_value_usd',                  # Target
            'exporter_gdp_per_capita_usd',      # Wealth/Development
            'importer_gdp_per_capita_usd',
            'exporter_logistics_index',         # Infrastructure Quality
            'importer_logistics_index',
            'logistics_index_gap',              # The infrastructure "distance"
            'exporter_temp_change_c',           # Climate Stability
            'importer_temp_change_c',
            'trade_value_usd_volatility',       # Economic Stability
            'gdp_exporter/importer',            # Imbalance Ratios
            'population_exporter/importer',
            'pc1_importer',                      # PCA Components
            'pc2_importer',
            'pc1_exporter',
            'pc2_exporter',
        ]
        plot_correlation_heatmap(df_eda, efficiency_features, title="Correlation Matrix of Trade Efficiency Features", log=True)

        features2 = [
            'trade_value_usd',
            'any_trade_agreement',
            'cu',
            'fta',
            'psa',
            'eia',
            'cueia',
            'ftaeia',
            'psaeia',
        ]
        plot_correlation_heatmap(df, features2, title="Correlation Matrix of Trade Agreements", log=True)

    if PIPELINE_CONFIG["return_df"]:
        return load_latest_eda_data()

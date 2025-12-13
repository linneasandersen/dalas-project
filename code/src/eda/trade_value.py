import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.config import PRODUCT_NAMES
import matplotlib.ticker as mtick

def histogram_trade_values(df: pd.DataFrame) -> None:
    """Plot histogram of trade values."""
    plt.figure(figsize=(10, 6))
    plt.hist(df['trade_value_usd'].dropna(), bins=100, color='blue', alpha=0.7)
    plt.title('Histogram of Trade Values (USD)')
    plt.xlabel('Trade Value (USD)')
    plt.ylabel('Frequency')
    plt.yscale('log') 
    plt.grid(True)
    plt.show()

def histogram_trade_values_log_x(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(np.log1p(df['trade_value_usd']), color='darkblue', bins=100)
    plt.title("Histogram of Log-Scaled Trade Values")
    plt.xlabel("log(1 + Trade Value)")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.grid(True)
    plt.show()

def histogram_trade_value_by_product(df: pd.DataFrame, hs2_code: int, color) -> None:
    """Plot histogram of trade values for a specific HS2 product code."""
    product_df = df[df['hs2'] == hs2_code]
    plt.figure(figsize=(10, 6))
    plt.hist(product_df['trade_value_usd'].dropna(), bins=100, color=color, alpha=0.7)
    plt.title(f'Histogram of Trade Values (USD) for {PRODUCT_NAMES.get(hs2_code, "Unknown Product")}')
    plt.xlabel('Trade Value (USD)')
    plt.ylabel('Frequency')
    plt.yscale('log') 
    plt.grid(True)
    plt.show()

def histogram_trade_value_by_product_log_x(df: pd.DataFrame, hs2_code: int, color) -> None:
    product_df = df[df['hs2'] == hs2_code]
    plt.figure(figsize=(10, 6))
    plt.hist(np.log1p(product_df['trade_value_usd']), color=color, bins=100)
    plt.title(f"Histogram of Log-Scaled Trade Values for {PRODUCT_NAMES.get(hs2_code, 'Unknown Product')}")
    plt.xlabel("log(1 + Trade Value)")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.grid(True)
    plt.show()

def countries_centered_plot(df: pd.DataFrame, country_column='exporter_name', country_index: int = 0, color='skyblue') -> None:
    """
    Visualize exporters with a given country at the center, using mean trade per year per country.
    
    Parameters:
        df: DataFrame with 'exporter_name', 'year', and 'trade_value_usd'.
        country_index: Index of the country to center the plot on (0 = highest exporter by mean trade).
    """
    # Compute mean trade per year per country
    mean_per_year = df.groupby([country_column, 'year'])['trade_value_usd'].mean().reset_index()
    
    # Compute overall mean per country
    agg = mean_per_year.groupby(country_column)['trade_value_usd'].mean().sort_values(ascending=False)
    
    # Get the country at the chosen index
    center_country = agg.index[country_index]
    center_value = agg.iloc[country_index]
    
    # Split countries into "more" and "less" than the center country
    more = agg[agg > center_value].sort_values(ascending=True)   # smaller on top
    less = agg[agg < center_value].sort_values(ascending=False)  # smaller on bottom
    
    # Combine for plotting
    plot_df = pd.concat([more, pd.Series({center_country: center_value}), less])
    
    # Plot
    plt.figure(figsize=(8, len(plot_df)*0.4))
    colors = [color]*len(more) + [color] + [color]*len(less)
    plt.barh(plot_df.index, plot_df.values, color=colors)
    plt.xlabel('Mean Trade Value per Year (USD)')
    plt.title(f"Importers by Mean Trade Value per Year")
    plt.gca().xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    plt.show()

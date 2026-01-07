import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.config import PRODUCT_NAMES, PRODUCT_PALETTE, PRODUCT_NAMES
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

# Cumulative share plot
def cumulative_share_plot(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 6))
    vals = df['trade_value_usd'].sort_values(ascending=False)
    cum_share = vals.cumsum() / vals.sum()

    plt.plot(cum_share.values)
    plt.xlabel("Top N trade flows")
    plt.ylabel("Cumulative share of trade")
    plt.title("Cumulative Share of Trade by Top N Trade Flows")
    plt.grid(True)
    plt.show()

def cumulative_share_plot_percent(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 6))
    vals = df['trade_value_usd'].sort_values(ascending=False)
    cum_share = vals.cumsum() / vals.sum()
    percent_trade_flows = np.arange(1, len(vals) + 1) / len(vals) * 100

    plt.plot(percent_trade_flows, cum_share.values)
    plt.xlabel("Percentage of Trade Flows (%)")
    plt.ylabel("Cumulative Share of Trade")
    plt.title("Cumulative Share of Trade by Percentage of Trade Flows")
    colors = {0.01: 'lightsteelblue', 0.05: 'cornflowerblue', 0.1: 'royalblue'}
    for p in [0.01, 0.05, 0.1]:
        n = int(p * len(vals))
        # annotate with different colors
        color=colors.get(p, 'grey')
        plt.axvline(x=p*100, color=color, linestyle='--', alpha=0.7, linewidth=1, label=f'{int(p*100)}% of trade flows')
        print(p*100, cum_share.iloc[n].round(4))
    # legend for the lines
    plt.legend()
    plt.grid(True)
    plt.show()


def trade_over_time(df, product=None):
    """
    Plot trade value over time
    """
    if product:
        df = df[df['hs2'] == product]
    
    trade_per_year = df.groupby('year')['trade_value_usd'].sum().reset_index()
    trade_per_year = trade_per_year.sort_values('year')

    top10_per_year = (
        df.groupby('year')
        .apply(lambda x: x['trade_value_usd'].nlargest(int(0.1*len(x))).sum())
        .reset_index(name='top10_trade')
    )

    bottom90_per_year = (
        df.groupby('year')
        .apply(lambda x: x['trade_value_usd'].nsmallest(int(0.9*len(x))).sum())
        .reset_index(name='bottom90_trade')
    )

    # Plotting
    plt.figure(figsize=(8, 6))

    # Raw trade values
    colors = {'10th': 'lightcoral', '50th': 'royalblue', '90th': 'seagreen', 'total': 'midnightblue'}
    plt.plot(trade_per_year['year'], trade_per_year['trade_value_usd'],
                 marker='o', color=colors['total'], label='Total Trade')
    plt.title('Total Trade Value Over Time')
    plt.xlabel('Year')
    plt.ylabel('Total Trade Value (USD)')
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    plt.grid(True)

    # add top90 and top10 lines
    plt.plot(bottom90_per_year['year'], bottom90_per_year['bottom90_trade'],
                 marker='o', label='Bottom 90% of trade', color=colors['10th'])
    plt.plot(top10_per_year['year'], top10_per_year['top10_trade'],
                 marker='o', label='Top 10% of trade', color=colors['90th'])
    plt.legend()

    plt.tight_layout()
    plt.show()


def trade_over_time_products(df, products, include_top_bottom=False, include_legend=True, title="Trade Value Over Time by Product"):
    """
    Plot trade value over time for a list of products.
    Bottom 90% and top 10% are shown with different linestyles and transparency.
    """
    plt.figure(figsize=(14, 6))

    for product in products:
        df_prod = df[df['hs2'] == product]
        
        trade_per_year = df_prod.groupby('year')['trade_value_usd'].sum().reset_index()
        trade_per_year = trade_per_year.sort_values('year')

        top10_per_year = (
            df_prod.groupby('year')
                   .apply(lambda x: x['trade_value_usd'].nlargest(int(0.1*len(x))).sum())
                   .reset_index(name='top10_trade')
        )

        bottom90_per_year = (
            df_prod.groupby('year')
                   .apply(lambda x: x['trade_value_usd'].nsmallest(int(0.9*len(x))).sum())
                   .reset_index(name='bottom90_trade')
        )

        color = PRODUCT_PALETTE[product]['primary']
        name = PRODUCT_NAMES.get(product, f'Product {product}')

        # Plot total trade
        plt.plot(trade_per_year['year'], trade_per_year['trade_value_usd'],
                 marker='o', color=color, label=f'{name} - Total trade')


        # Bottom 90% (dashed, semi-transparent)
        if include_top_bottom:
            plt.plot(bottom90_per_year['year'], bottom90_per_year['bottom90_trade'],
                     linestyle='--', color=color, alpha=0.5,
                     label=f'{name} - Bottom 90% of trade')

            # Top 10% (dotted, semi-transparent)

            plt.plot(top10_per_year['year'], top10_per_year['top10_trade'],
                     linestyle=':', color=color, alpha=0.8,
                     label=f'{name} - Top 10% of trade')

    plt.title(title)
    plt.xlabel('Year')
    plt.ylabel('Trade Value (USD)')
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    plt.grid(True)
    if include_legend:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    plt.tight_layout()
    plt.show()


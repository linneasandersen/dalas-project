import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import PRODUCT_NAMES_SHORT, PRODUCT_PALETTE


def boxplot_trade_value_by_product(df: pd.DataFrame, log=True) -> None:
    # 1. Prepare Data
    plot_df = df.copy()
    val_col = "log_trade_value" if log else "trade_value_usd"
    
    if log:
        plot_df[val_col] = np.log1p(plot_df["trade_value_usd"])

    # 2. Map Names and Colors
    # Using PRODUCT_NAMES_SHORT for cleaner labels
    plot_df["product_name"] = plot_df["hs2"].map(PRODUCT_NAMES_SHORT)
    
    # Extract 'primary' hex codes for the palette
    custom_palette = {PRODUCT_NAMES_SHORT[code]: info["primary"] 
                     for code, info in PRODUCT_PALETTE.items()}

    # 3. Determine Order (by median)
    order = (
        plot_df.groupby("product_name")[val_col]
        .median()
        .sort_values()
        .index
    )

    # 4. Create Plot
    plt.figure(figsize=(12, 8))
    
    sns.boxplot(
        data=plot_df,
        x=val_col,
        y="product_name", # Horizontal boxplot often easier to read with long names
        order=order,
        palette=custom_palette,
        showfliers=False,
        linewidth=1.5
    )

    # 5. Styling
    plt.title("Distribution of Trade Value by Product Category", fontsize=14, pad=20)
    plt.xlabel("Log Trade Value (log(1 + USD))" if log else "Trade Value (USD)", fontsize=12)
    plt.ylabel("") # Removed to save space since names are self-explanatory
    
    sns.despine() # Clean up the chart borders
    plt.tight_layout()
    plt.show()

def boxplot_raw_trade_value(df, trade_col="trade_value_usd"):
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=df[trade_col], showfliers=False)
    plt.yscale("log")
    plt.ylabel("Trade Value (USD)")
    plt.title("Boxplot of Raw Trade Values (Log Scale)")
    plt.tight_layout()
    plt.show()

def boxplot_trade_value_all(df: pd.DataFrame, log=True) -> None:
    print("Generating boxplot for all trade values...")
    # 1. Prepare Data
    plot_df = df.copy()
    val_col = "log_trade_value" if log else "trade_value_usd"
    
    if log:
        plot_df[val_col] = np.log1p(plot_df["trade_value_usd"])

    # 4. Create Plot
    plt.figure(figsize=(12, 8))
    
    sns.boxplot(
        data=plot_df,
        x=val_col,
        color='darkblue' if log else 'cornflowerblue',
        showfliers=False,
        linewidth=1.5,
        width=0.25,
        medianprops={"color": "lightgrey", "linewidth": 1.5} if log else None
    )

    # 5. Styling
    plt.title("Distribution of Trade Value", fontsize=14, pad=20)
    plt.xlabel("Log Trade Value (log(1 + USD))" if log else "Trade Value (USD)", fontsize=12)
    plt.ylabel("") 
    
    sns.despine() # Clean up the chart borders
    plt.tight_layout()
    plt.show()
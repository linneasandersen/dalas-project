
from src.config import GOOGLE_DRIVE
from src.eda.sanity_check import check_sanity
from src.eda.trade_value import countries_centered_plot, histogram_trade_value_by_product_log_x, histogram_trade_values, histogram_trade_values_log_x, histogram_trade_value_by_product


PIPELINE_CONFIG = {
    "sanity_check": False,
    "trade_value_histograms": False,
    "trade_value": True
}

# ---------------------------------------------------
# 3. Main orchestration
# ---------------------------------------------------

def explore(df):
    processed_dir = GOOGLE_DRIVE / "processed"
    return_df = None

    if PIPELINE_CONFIG["sanity_check"]:
        check_sanity(df)
        
    if PIPELINE_CONFIG["trade_value_histograms"]:
        histogram_trade_values(df)
        histogram_trade_values_log_x(df)
        histogram_trade_value_by_product(df, hs2_code=15, color='sandybrown')
        histogram_trade_value_by_product_log_x(df, hs2_code=15, color='sienna')

    if PIPELINE_CONFIG["trade_value"]:
        #countries_centered_plot(df, 0)
        countries_centered_plot(df, country_column='importer_name', country_index=0, color='lightcoral')



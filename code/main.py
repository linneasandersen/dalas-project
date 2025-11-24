from src.processing.hs import update_config
from src.processing.pipelines import process_dataframe
from src.fetch.hs import fetch_hs
from src.fetch.countries import fetch_countries
from src.fetch.oec import fetch_oec_trade, load_oec_trade
from src.fetch.fao import fetch_FAO
from src.processing.clean import clean_all
from src.utils.io import save_df

def main():
    #countries = fetch_countries()
    #print(countries.head())

    
    print("Fetching OEC trade data...")
    oec = fetch_oec_trade()
    print(oec.head())
    save_df(oec, "oec_trade_raw_all")

    oec = load_oec_trade()
    print(oec.head())

    to_clean = [oec]
    

    # forest, production, temp_change = fetch_FAO()
    # print(forest.head())
    # print(production.head())
    # print(temp_change.head())

    
    # hs = fetch_hs()

    # # list of all dfs
    # to_clean = [hs]
    cleaned = clean_all(to_clean)
    # hs = cleaned[0]
    oec = cleaned[0]
    
    # print(hs.head())

    # # process dataframes
    # hs = process_dataframe("hs", hs)
    oec = process_dataframe("oec", oec)
    print(oec.head())
    save_df(oec, "oec_trade_filter_countries")

    # save_df(hs, "hs2_filter_products")

    # print(update_config(hs))
    
    #for name, df in cleaned.items():
    #    save_df(df, f"cleaned_{name}")

if __name__ == "__main__":
    main()

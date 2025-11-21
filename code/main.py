from src.processing.pipelines import process_dataframe
from src.fetch.hs import fetch_hs
from src.fetch.countries import fetch_countries
from src.fetch.oec import fetch_oec_trade
from src.fetch.fao import fetch_FAO
from src.processing.clean import clean_all
from src.utils.io import save_df

def main():
    #countries = fetch_countries()
    #print(countries.head())

    print("Fetching OEC trade data...")
    #oec = fetch_oec_trade()
    #print(oec.head())
    #save_df(oec, "oec_trade_raw")

    # forest, production, temp_change = fetch_FAO()
    # print(forest.head())
    # print(production.head())
    # print(temp_change.head())

    hs = fetch_hs()
    print(hs.head())

    # list of all dfs
    #cleaned = [countries, oec, forest, production, temp_change, hs]

    # process dataframes
    hs = process_dataframe("hs", hs)

    #for name, df in cleaned.items():
    #    save_df(df, f"cleaned_{name}")

if __name__ == "__main__":
    main()

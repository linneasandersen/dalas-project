import certifi
import pandas as pd
from src.config import RAW_DIR, OEC_YEARS
from pathlib import Path
import requests

def fetch_oec_trade():
    # Iterate and fetch export data
    df = []
    url = "https://api-v2.oec.world/tesseract/data.jsonrecords"
    years = OEC_YEARS

    for year in years:
        print(f"Fetching OEC export data for year {year}...")
        params = {
            "cube": "trade_i_baci_a_92",
            "locale": "en",
            "drilldowns": "Exporter Country,Importer Country,Year,HS2",
            "measures": "Trade Value",
            "include": f"Year:{year}"
        }
        r = requests.get(
            url,
            params=params,
            verify=certifi.where()
        )
        try:
            df_year = pd.DataFrame(r.json()["data"])
            df.append(df_year)
        except KeyError:
            print(f"No data found for export in {year}")


    # concatenate
    df = pd.concat(df)
    return df



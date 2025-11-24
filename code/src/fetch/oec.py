from itertools import product
from time import time
import certifi
import pandas as pd
from src.config import PROCESSED_DIR, YEARS, OEC_CODES
from pathlib import Path
import requests


import requests
import pandas as pd
import certifi
import time

from itertools import product
from time import sleep
import certifi
import pandas as pd
from pathlib import Path
import requests
from src.config import PROCESSED_DIR, YEARS, OEC_CODES

# place log file in current directory, i.e code directory
LOG_FILE = LOG_FILE = Path("oec_fetch_log.csv")

def fetch_oec_trade(limit=1000):
    url = "https://api-v2.oec.world/tesseract/data.jsonrecords"
    df_list = []
    hs2_ids = OEC_CODES
    years = YEARS

    # Ensure log file exists
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("HS2,Year,Offset,Status\n")  # header

    for hs2 in hs2_ids:
        for year in years:
            offset = 0
            while True:
                print(f"Fetching HS2: {hs2}, Year: {year}, Offset: {offset}...")
                params = {
                    "cube": "trade_i_baci_a_92",
                    "drilldowns": "HS2,Exporter Country,Importer Country,Year",
                    "measures": "Trade Value",
                    "include": f"HS2:{hs2};Year:{year}",
                    "limit": f"{limit},{offset}"
                }

                try:
                    r = requests.get(url, params=params, verify=certifi.where(), timeout=30)
                    print(r.url)
                    if r.status_code != 200:
                        print(f"HTTP {r.status_code}, skipping this batch")
                        # log failed batch
                        with open(LOG_FILE, "a") as f:
                            f.write(f"{hs2},{year},{offset},HTTP {r.status_code}\n")
                        break

                    data = r.json().get("data", [])
                    if not data:
                        print("No more data for this HS2-year")
                        # log empty batch
                        with open(LOG_FILE, "a") as f:
                            f.write(f"{hs2},{year},{offset},No data\n")
                        break

                    df_list.append(pd.DataFrame(data))
                    offset += limit
                    sleep(1)  # polite delay

                except requests.exceptions.RequestException as e:
                    print(f"Request failed: {e}, retrying in 5s")
                    # log exception
                    with open(LOG_FILE, "a") as f:
                        f.write(f"{hs2},{year},{offset},Exception: {e}\n")
                    sleep(5)

    if df_list:
        df = pd.concat(df_list, ignore_index=True)
        return df
    else:
        print("No data fetched")
        return pd.DataFrame()


def fetch_oec_trade_no_log(limit=1000):
    url = "https://api-v2.oec.world/tesseract/data.jsonrecords"
    df_list = []
    hs2_ids = OEC_CODES
    years = YEARS

    for hs2 in hs2_ids:
        for year in years:
            offset = 0
            while True:
                print(f"Fetching HS2: {hs2}, Year: {year}, Offset: {offset}...")
                params = {
                    "cube": "trade_i_baci_a_92",
                    "drilldowns": "HS2,Exporter Country,Importer Country,Year",
                    "measures": "Trade Value",
                    "include": f"HS2:{hs2};Year:{year}",
                    "limit": f"{limit},{offset}"
                }

                try:
                    r = requests.get(url, params=params, verify=certifi.where(), timeout=30)
                    print(r.url)
                    if r.status_code != 200:
                        print(f"HTTP {r.status_code}, skipping this batch")
                        break

                    data = r.json().get("data", [])
                    if not data:
                        print("No more data for this HS2-year")
                        break

                    df_list.append(pd.DataFrame(data))
                    offset += limit
                    time.sleep(1)  # be nice to the API

                except requests.exceptions.RequestException as e:
                    print(f"Request failed: {e}, retrying in 5s")
                    time.sleep(5)

    if df_list:
        df = pd.concat(df_list, ignore_index=True)
        return df
    else:
        print("No data fetched")
        return pd.DataFrame()


def fetch_oec_trade_old2():
    df_list = []
    url = "https://api-v2.oec.world/tesseract/data.jsonrecords"

    countries = ["euesp", "eufra", "euitl", "afago", "afben"]
    hs2_ids = [101, 102]  # your product list
    years = [2020, 2021]

    # All combinations of exporter, importer, HS2, year
    combinations = [
        (exp, imp, hs2, year)
        for exp, imp, hs2, year in product(countries, countries, hs2_ids, years)
        if exp != imp
    ]

    for exporter, importer, hs2, year in combinations:
        print(f"Fetching OEC trade data for {exporter} to {importer}, HS2: {hs2}, Year: {year}...")
        params = {
            "cube": "trade_i_baci_a_22",
            "drilldowns": "HS2,Exporter Country,Importer Country,Year",
            "measures": "Trade Value",
            "include": f"HS2:{hs2};Exporter+Country:{exporter};Importer+Country:{importer};Year:{year}",
        }

        r = requests.get(
            url,
            params=params,
            verify=certifi.where()
        )
        if r.status_code != 200:
            print(f"HTTP {r.status_code} for year {year}")
            print("URL:", r.url)
            continue

        # JSON decoding error?
        try:
            json_data = r.json()
        except ValueError:
            print(f"Non-JSON response for year {year}")
            print("URL:", r.url)
            print("Response snippet:", r.text[:500])
            continue

        # No 'data' key or empty data
        data = json_data.get("data")
        if not data:
            print(f"No data returned for year {year}")
            continue

        df_list.append(pd.DataFrame(data))
        time.sleep(1)

    # concatenate
    df = pd.concat(df_list)
    return df
    # handle response as before




def fetch_oec_trade_old():
    # Iterate and fetch export data
    df_list = []
    url = "https://api-v2.oec.world/tesseract/data.jsonrecords"
    years = YEARS
    products = PRODUCT_CODES
    LIMIT = 1000  # max records per request

    for year in years:
        offset = 0
        #for product in products:
        print(f"Fetching OEC export data for year {year} with offset {offset}...")
        params = {
            "cube": "trade_i_baci_a_92",        
            "locale": "en",
            "drilldowns": "Exporter Country,Importer Country,Year,HS2",
            "measures": "Trade Value",
            "Year": year,
            "limit": f"{LIMIT},{offset}"
        }
        r = requests.get(
            url,
            params=params,
            verify=certifi.where()
        )
        if r.status_code != 200:
            print(f"HTTP {r.status_code} for year {year}")
            print("URL:", r.url)
            continue

        # JSON decoding error?
        try:
            json_data = r.json()
        except ValueError:
            print(f"Non-JSON response for year {year}")
            print("URL:", r.url)
            print("Response snippet:", r.text[:500])
            continue

        # No 'data' key or empty data
        data = json_data.get("data")
        if not data:
            print(f"No data returned for year {year}")
            continue

        offset += LIMIT
        df_list.append(pd.DataFrame(data))
        time.sleep(1.2)

    # concatenate
    df = pd.concat(df_list)
    return df


def load_oec_trade():
    path = PROCESSED_DIR / "oec_trade_raw_all.csv"
    df = pd.read_csv(path)
    return df
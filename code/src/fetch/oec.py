from importlib.resources import path
from itertools import product
from pathlib import Path
from time import sleep
import certifi
import pandas as pd
import requests
from datetime import datetime
import zipfile

from src.config import PROCESSED_DIR, YEARS, OEC_CODES, RAW_DIR, LOCAL_DIR

def fetch_oec_trade_file(year, dir=LOCAL_DIR, filename=None):
    if filename is None:
        filename = f"trade_i_baci_a_92_{year}.csv"
    path = dir / filename
    print(path)
    # unzip file if not already unzipped
    if not path.exists():
        zip_path = dir / f"{filename}.zip"
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dir)
    df = pd.read_csv(path)
    return df

def fetch_oec_trade_all(dir=LOCAL_DIR, filename=None):
    if filename is None:
        filename = "oec_trade_filtered_all.csv"
    path = dir / filename
    df = pd.read_csv(path)
    return df

def delete_oec_trade_file(year, dir=LOCAL_DIR):
    path = dir / f"trade_i_baci_a_92_{year}.csv"
    if path.exists():
        path.unlink()
        print(f"Deleted {path}")
    else:
        print(f"File {path} does not exist.")


def fetch_oec_trade(year, limit=1000):
    url = "https://api-v2.oec.world/tesseract/data.jsonrecords"
    df_list = []
    hs2_ids = OEC_CODES

    # 1. Create a UNIQUE log file for this specific run
    # This prevents overwriting your previous logs if you run the script twice.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_log_file = Path(f"oec_fetch_log_{timestamp}.csv")
    
    # Write header
    current_log_file.write_text("HS2,Year,Offset,Status,Details\n")
    print(f"Logging to: {current_log_file}")

    for hs2 in hs2_ids:
        offset = 0
        
        # Loop for offsets (0, 1000, 2000...)
        while True:
            print(f"Fetching HS2: {hs2}, Year: {year}, Offset: {offset}...")
            
            params = {
                "cube": "trade_i_baci_a_92",
                "drilldowns": "Exporter Country,Importer Country,Year, HS2",
                "include": f"HS2:{hs2};Year:{year}",
                "limit": f"{limit},{offset}"
            }

            retries = 0
            max_retries = 3
            batch_success = False
            finished_hs2 = False

            # Retry Loop
            while retries < max_retries:
                try:
                    # 2. Increased timeout to 60s
                    r = requests.get(url, params=params, verify=certifi.where(), timeout=60)
                    print(r.url)

                    # --- Scenario A: Server Busy (503/504) ---
                    if r.status_code in [500, 502, 503, 504]:
                        print(f"  > Server Busy (HTTP {r.status_code}). Retrying in 10s... ({retries+1}/{max_retries})")
                        retries += 1
                        sleep(10) # Give server a break
                        continue 

                    # --- Scenario B: Client Error (404/400) ---
                    if r.status_code != 200:
                        print(f"  > HTTP {r.status_code}. Skipping this batch.")
                        with open(current_log_file, "a") as f:
                            f.write(f"{hs2},{year},{offset},HTTP_ERROR,{r.status_code}\n")
                        break # Break retry loop, effectively skipping this offset

                    # --- Scenario C: Success ---
                    data = r.json().get("data", [])
                    
                    if not data:
                        print("  > No more data for this HS2 code.")
                        with open(current_log_file, "a") as f:
                            f.write(f"{hs2},{year},{offset},FINISHED,No Data\n")
                        finished_hs2 = True # Flag to break the OUTER loop
                        break 
                    
                    # Data found
                    df_list.append(pd.DataFrame(data))
                    batch_success = True
                    sleep(1) # Polite delay
                    break # Success! Break the retry loop

                except requests.exceptions.RequestException as e:
                    print(f"  > Connection Error: {e}. Retrying in 5s... ({retries+1}/{max_retries})")
                    with open(current_log_file, "a") as f:
                        f.write(f"{hs2},{year},{offset},EXCEPTION,{str(e)}\n")
                    retries += 1
                    sleep(5)

            # --- Decisions after Retry Loop ---

            if finished_hs2:
                # We reached the end of data for this HS2 code
                break

            if batch_success:
                # Normal success, move to next page
                offset += limit
            else:
                # We exhausted retries OR had a 4xx error.
                # CRITICAL FIX: We do NOT break the outer loop. We SKIP this batch.
                print(f"  >>> FAILED BATCH after retries. Skipping offset {offset} -> {offset+limit}")
                with open(current_log_file, "a") as f:
                    f.write(f"{hs2},{year},{offset},SKIPPED_BATCH,Max Retries/Error\n")
                
                offset += limit # Move forward anyway to try to get the next chunk

    if df_list:
        df = pd.concat(df_list, ignore_index=True)
        return df
    else:
        print("No data fetched")
        return pd.DataFrame()
    
    
def fetch_oec_trade_oldlog(year, limit=1000):
    url = "https://api-v2.oec.world/tesseract/data.jsonrecords"
    df_list = []
    hs2_ids = OEC_CODES

    # Ensure log file exists
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("HS2,Year,Offset,Status\n")  # header

    for hs2 in hs2_ids:
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
                r = requests.get(url, params=params, verify=certifi.where(), timeout=60)
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
                    sleep(1)  # be nice to the API

                except requests.exceptions.RequestException as e:
                    print(f"Request failed: {e}, retrying in 5s")
                    sleep(5)

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
        sleep(1)

    # concatenate
    df = pd.concat(df_list)
    return df


def load_oec_trade():
    path = PROCESSED_DIR / "oec_trade_raw_all.csv"
    df = pd.read_csv(path)
    return df
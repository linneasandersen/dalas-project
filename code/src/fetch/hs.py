import pandas as pd
from src.config import RAW_DIR, RAW_DIR_EN

def fetch_hs():
    # check which directory to use
    path = RAW_DIR_EN / "hs_product_classification.csv"
    if path.exists():
        print(path)
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(RAW_DIR / "hs_product_classification.csv", sep=';', on_bad_lines='skip')
    return df

def fetch_hs2cpc_mapping():
    path = RAW_DIR / "H2_to_CP.csv"
    if path.exists():
        print(path)
        df = pd.read_csv(path, encoding='latin1')
    else:
        df = pd.read_csv(RAW_DIR / "H2_to_CP.csv", sep=';', on_bad_lines='skip')
    return df

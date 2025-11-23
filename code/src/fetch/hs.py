import pandas as pd
from src.config import RAW_DIR, RAW_DIR_EN

def fetch_hs():
    # check which directory to use
    path = RAW_DIR_EN / "hs_product_classification.csv"
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(RAW_DIR / "hs_product_classification.csv")
    return df

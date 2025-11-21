import pandas as pd
from src.config import PROCESSED_DIR, RAW_DIR

def fetch_hs():
    df = pd.read_csv(RAW_DIR / "hs_product_classification.csv")
    return df

import pandas as pd
from src.config import PROCESSED_DIR, RAW_DIR

def fetch_FAO():
    forest = pd.read_csv(PROCESSED_DIR / "fao_forest_data.csv")
    production = pd.read_csv(PROCESSED_DIR / "fao_prod_data.csv")
    temp_change = pd.read_csv(PROCESSED_DIR / "fao_env_data.csv")   

    return forest, production, temp_change


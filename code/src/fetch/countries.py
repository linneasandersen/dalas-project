import pandas as pd
from pathlib import Path
from src.config import RAW_DIR
import os

def fetch_countries():
    # Countries from kaggle
    from kagglehub import kagglehub

    path = kagglehub.dataset_download("emolodov/country-codes-alpha2-alpha3")

    filename = "Country Codes Alpha-2 Alpha-3.csv"
    file_path = os.path.join(path, filename)

    return pd.read_csv(file_path)


from pathlib import Path
from src.config import PROCESSED_DIR

def save_df(df, name: str):
    path = PROCESSED_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    return path

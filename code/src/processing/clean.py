import pandas as pd
from typing import List

def clean_all(dfs: List[pd.DataFrame]):
    df_cleaned = []
    for df in dfs:
        df.columns = df.columns.str.lower()
        df_cleaned.append(df)

    return df_cleaned
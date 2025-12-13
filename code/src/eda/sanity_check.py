import pandas as pd

def check_sanity(df: pd.DataFrame) -> None:
    """Perform sanity checks on the DataFrame."""
    df.duplicated(['exporter_id','importer_id','hs2','year']).sum()
    print(f"Number of duplicate rows based on exporter_id, importer_id, hs2, year: {df.duplicated(['exporter_id','importer_id','hs2','year']).sum()}")

    df.isna().mean().sort_values(ascending=False)
    print("Proportion of missing values per column:")
    print(df.isna().mean().sort_values(ascending=False))
    
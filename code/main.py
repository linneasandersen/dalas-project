from src.data.main import build
from src.eda.main import explore

if __name__ == "__main__":
    df = build()
    explore(df)

# TODO: clean other FAO datasets and process them
# TODO: merge FAO datasets with primary dataset
# TODO: merge with tariff data
# TODO: try baseline model with new columns
# TODO: try baseline model with per product dataset


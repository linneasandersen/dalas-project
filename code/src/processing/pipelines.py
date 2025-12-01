from src.processing.hs import process_hs
from src.processing.oec import process_oec
from src.processing.fao import process_forest, process_production, process_temp_change

pipelines = {
    "hs": process_hs,
    "oec": process_oec,
    "fao_forest": process_forest,
    "fao_production": process_production,
    "fao_temp_change": process_temp_change,
}

def process_dataframe(name, df):
    return pipelines[name](df)

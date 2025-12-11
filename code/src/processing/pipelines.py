from src.processing.hs import process_hs, process_cpc
from src.processing.oec import process_oec
from src.processing.fao import process_forest, process_production, process_temp_change
from src.processing.country_meta import process_gdp, process_land, process_population

pipelines = {
    "hs": process_hs,
    "hs2cpc_mapping": process_cpc,
    "oec": process_oec,
    "fao_forest": process_forest,
    "fao_production": process_production,
    "fao_temp_change": process_temp_change,
    "gdp": process_gdp,
    "land": process_land,
    "population": process_population,
}

def process_dataframe(name, df):
    return pipelines[name](df)

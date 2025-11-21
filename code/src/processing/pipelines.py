from src.processing.hs import process_hs

pipelines = {
    "hs": process_hs,
}

def process_dataframe(name, df):
    return pipelines[name](df)

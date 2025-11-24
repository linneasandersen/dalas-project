from src.processing.hs import process_hs

pipelines = {
    "hs": process_hs,
    "oec": process_oec,
}

def process_dataframe(name, df):
    return pipelines[name](df)

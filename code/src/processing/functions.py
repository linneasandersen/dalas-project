

def filter_data(df, column, values):
    return df[df[column].isin(values)]
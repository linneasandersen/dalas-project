import pandas as pd

def interpolate_logistics_index(df, exporter_col='exporter_name', importer_col='importer_name',
                                exporter_index_col='exporter_logistics_index',
                                importer_index_col='importer_logistics_index'):
    """
    Interpolates missing logistics index values linearly per country.
    
    Parameters:
    - df: pandas DataFrame containing trade data
    - exporter_col: column name for exporter country ID
    - importer_col: column name for importer country ID
    - exporter_index_col: column name for exporter logistics index
    - importer_index_col: column name for importer logistics index
    
    Returns:
    - df: DataFrame with interpolated logistics index columns
    """
    print("Interpolating missing logistics index values...")
    print(df.columns)

    # count missing values before interpolation
    missing_exporter_before = df[exporter_index_col].isnull().sum()
    missing_importer_before = df[importer_index_col].isnull().sum()
    print(f"Missing exporter logistics index before interpolation: {missing_exporter_before}")
    print(f"Missing importer logistics index before interpolation: {missing_importer_before}")
    
    # Interpolate exporter logistics index per country
    df[exporter_index_col] = df.groupby(exporter_col, group_keys=False)[exporter_index_col].apply(
        lambda x: x.interpolate(method='linear')
    )
    
    # Interpolate importer logistics index per country
    df[importer_index_col] = df.groupby(importer_col, group_keys=False)[importer_index_col].apply(
        lambda x: x.interpolate(method='linear')
    )

    # count missing values after interpolation
    missing_exporter_after = df[exporter_index_col].isnull().sum()
    missing_importer_after = df[importer_index_col].isnull().sum()
    print(f"Missing exporter logistics index after interpolation: {missing_exporter_after}")
    print(f"Missing importer logistics index after interpolation: {missing_importer_after}")

    df['exporter_logistics_index'] = df.groupby('exporter_name', group_keys=False)['exporter_logistics_index'].apply(lambda x: x.fillna(x.mean()))
    df['importer_logistics_index'] = df.groupby('importer_name', group_keys=False)['importer_logistics_index'].apply(lambda x: x.fillna(x.mean()))

    # final count of missing values
    missing_exporter_final = df[exporter_index_col].isnull().sum()
    missing_importer_final = df[importer_index_col].isnull().sum()
    print(f"Missing exporter logistics index after filling with mean: {missing_exporter_final}")
    print(f"Missing importer logistics index after filling with mean: {missing_importer_final}")

    return df

import numpy as np
from sklearn.model_selection import train_test_split

# ---------------------------------------------------
# Panel splits
# ---------------------------------------------------

def panel_train_val_test_split(df, time_col='year', last_test_year=2023, val_years=2):
    df = df.sort_values(time_col).reset_index(drop=True)

    test_df = df[df[time_col] == last_test_year]

    val_years_range = range(last_test_year - val_years, last_test_year)
    val_df = df[df[time_col].isin(val_years_range)]

    train_df = df[~df[time_col].isin(list(val_years_range) + [last_test_year])]

    return train_df, val_df, test_df

def random_train_val_test_split(df, train_frac=0.7, val_frac=0.15, test_frac=0.15,random_state=42):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-8

    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - train_frac),
        random_state=random_state,
        shuffle=True,
    )

    # Second split: val vs test
    val_size = test_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=val_size,
        random_state=random_state,
        shuffle=True,
    )

    return train_df, val_df, test_df



def rolling_panel_split(df, time_col='year', last_test_year=2023, initial_train_years=10, val_window=1):
    df = df.sort_values(time_col).reset_index(drop=True)

    test_df = df[df[time_col] == last_test_year]
    train_val_df = df[df[time_col] < last_test_year]

    years = sorted(train_val_df[time_col].unique())

    splits = []
    start_idx = 0
    while start_idx + initial_train_years + val_window <= len(years):
        train_years = years[start_idx:start_idx + initial_train_years]
        val_years = years[start_idx + initial_train_years : start_idx + initial_train_years + val_window]

        train_df = train_val_df[train_val_df[time_col].isin(train_years)]
        val_df = train_val_df[train_val_df[time_col].isin(val_years)]

        splits.append((train_df, val_df))
        start_idx += 1

    return splits, test_df


#different splitting for the test set. In addition to a random split, you could also leave out some countries and the last year of data to further test generalisation. Not mandatory though.


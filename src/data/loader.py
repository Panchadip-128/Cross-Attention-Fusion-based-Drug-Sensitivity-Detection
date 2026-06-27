import pandas as pd

def load_and_merge_gdsc(path_gdsc1: str, path_gdsc2: str) -> pd.DataFrame:
    """
    Load GDSC1 and GDSC2 CSV files, drop irrelevant columns, and merge.
    """
    gdsc1 = pd.read_csv(path_gdsc1)
    gdsc2 = pd.read_csv(path_gdsc2)

    gdsc1['source'] = 'GDSC1'
    gdsc2['source'] = 'GDSC2'

    cols_to_drop = ['log_max_conc_tested', 'log_max_conc_tested_2']
    gdsc1.drop(columns=[c for c in cols_to_drop if c in gdsc1.columns], inplace=True)
    gdsc2.drop(columns=[c for c in cols_to_drop if c in gdsc2.columns], inplace=True)

    df = pd.concat([gdsc1, gdsc2], ignore_index=True)
    return df

'''
Handling Specific File formats
'''

import pandas as pd


def read_mad(path):
    '''
    Read and pre-process a MAD file into a pandas Dataframe

    Args:
        path: Filepath to the MAD file
    '''
    df = pd.read_csv(path, delimiter='\t')
    df = clean_colnames(df)
    df = df.rename({'sample': 'limsid', '%cv': 'cv', "hemoglobin_mg/dl": "hemoglobin"}, axis=1)
    
    numeric_cols = ['dil_factor', 'mean_final_conc', 'sd', 'cv', 'reported_value',
                    'lloq', 'uloq', 'reportable_range_low', 'reportable_range_high', 'hemoglobin']
    
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
#     df['run_timestamp'] = pd.to_datetime(df['run_timestamp'], utc=True)
    
    return df

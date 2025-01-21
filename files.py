'''
Handling Specific File formats
'''

import re
import pandas as pd
from .pandas import clean_colnames


def read_mad(path):
    '''
    Read and pre-process a MAD file into a pandas Dataframe

    Args:
        path: Filepath to the MAD file
    '''
    df = pd.read_csv(path, delimiter='\t', dtype=object)
    df = clean_colnames(df)
    df = df.rename({'sample': 'limsid', '%cv': 'cv', 
                    'hemoglobin_mg/dl': 'hemoglobin',
                    'reported_value': 'result'}, axis=1)
    
    numeric_cols = ['dil_factor', 'mean_final_conc', 'sd', 'cv',
                    'lloq', 'uloq', 'reportable_range_low',
                    'reportable_range_high', 'hemoglobin']
    
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='ignore')
    df['result_numeric'] = pd.to_numeric(df.result, errors='coerce')

    df['result_filled'] = df.result
    df['is_filled'] = False
    df.loc[df.result.isin(['< LLOQ', '> ULOQ', 'Invalid']), "is_filled"] = True 
    df.loc[df.result.isin(['Invalid', '> ULOQ']), 'result_filled'] = df.mean_final_conc
    df.loc[df.result_filled == "< LLOQ", 'result_filled'] = df.reportable_range_low
    df['result_filled'] = pd.to_numeric(df.result_filled, errors='coerce')


    
    df['run_timestamp'] = df['run_timestamp'].str.replace(r"(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}) (?:.{1,10})", r'\1', regex=True)
    df['run_timestamp'] = pd.to_datetime(df['run_timestamp'])
    
    return df


def mad_add_loqs(df, g, uloq=True, lloq=True, log=True, rotate=0):
    loqs = df.groupby('assay').agg({'reportable_range_high': 'min', 'reportable_range_low': 'max'})
    axes = g.axes.flatten()
    
    if rotate != 0:
        g.set_xticklabels(rotation=rotate)
        g.fig.tight_layout()
    
    if log:
        for ax in g.fig.axes:
            ax.set_yscale('log')

    if uloq or lloq:
        for i, ax in enumerate(axes):
            if uloq:
                uloq_level = loqs.loc[re.search(r'=\s(.*)$', ax.get_title()).group(1), 'reportable_range_high']
                ax.axhline(uloq_level, ls='--', c='black')

            if lloq:
                lloq_level = loqs.loc[re.search(r'=\s(.*)$', ax.get_title()).group(1), 'reportable_range_low']
                ax.axhline(lloq_level, ls='--', c='black')


def read_lims(path, sheet_name=False):
    '''
    Read and pre-process a LIMS file into a pandas Dataframe

    Args:
        path: Filepath to the LIMS file
    '''
    if sheet_name:
        df = pd.read_excel(path, sheet_name='StarLIMS', dtype=object)
    else:
        df = pd.read_excel(path)

    df = clean_colnames(df)
    rename_cols_dict = {'sample_#': 'limsid', 'project_#': 'project', 
                        'sample_id': 'barcode', 'patient_id': 'subject',
                        'visit_code': 'visit', 
                        'sample_collection_timestamp': 'colld'}

    df = df.rename(rename_cols_dict, axis=1)

    date_cols = ['colld', 'received_on:']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])


    return df




def read_olink(path, npx=True, delimiter=";"):
    '''
    Read and pre-process an Olink file into a pandas Dataframe

    Args:
        path: Filepath to the Olink file
    '''

    df = pd.read_csv(path, delimiter=delimiter, dtype=object)
    df = clean_colnames(df)

    df['missingfreq'] = df.missingfreq.str.replace('%', '')

    if npx:
        df['result'] = df.npx

    numeric_cols = ["maxlod", "platelod", "result", "qc_deviation_inc_ctrl",
                      "qc_deviation_det_ctrl", "missingfreq"]

    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df['missingfreq'] = df.missingfreq / 100

    df.loc[:, "type"] = "sample"
    df.loc[df.sampleid.str.contains(r"^Neg"), "type"] = "negative"
    df.loc[df.sampleid.str.contains(r"^CA.*"), "type"] = "calibrator"
    df.loc[df.sampleid.str.contains(r"^IPC.*"), "type"] = "ipc"
    df.loc[df.sampleid.str.contains(r"^CS.*"), "type"] = "control"

    return df



def olink_add_loqs(df, g, uloq=True, lloq=True, log=True, rotate=0, npx=True):
    if npx:
        loqs = df.groupby('assay').agg({'maxlod': 'max'})

    axes = g.axes.flatten()
    
    if rotate != 0:
        g.set_xticklabels(rotation=rotate)
        g.fig.tight_layout()
    
    if log:
        for ax in g.fig.axes:
            ax.set_yscale('log')

    if npx:
        for i, ax in enumerate(axes):

            if lloq:
                lloq_level = loqs.loc[re.search(r'=\s(.*)$', ax.get_title()).group(1), 'maxlod']
                ax.axhline(lloq_level, ls='--', c='black')
    else:
        pass
        # if uloq or lloq:
        #     for i, ax in enumerate(axes):
        #         if uloq:
        #             uloq = loqs.loc[re.search(r'=\s(.*)$', ax.get_title()).group(1), 'reportable_range_high']
        #             ax.axhline(uloq, ls='--', c='black')

        #         if lloq:
        #             lloq = loqs.loc[re.search(r'=\s(.*)$', ax.get_title()).group(1), 'reportable_range_low']
        #             ax.axhline(lloq, ls='--', c='black')


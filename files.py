'''
Handling Specific File formats
'''

import re
import pandas as pd
import numpy as np
from IPython.display import display
from .pandas import clean_colnames
from .pandas import concat_cols
from .pandas import xv

##############################
# LIMS
##############################


def read_lims(path, sheet_name=False, subject_col='patient_id',
              collection=['subject', 'visit']):
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
                        'sample_id': 'barcode', subject_col: 'subject',
                        'visit_code': 'visit', 
                        'sample_collection_timestamp': 'coldt'}

    df = df.rename(rename_cols_dict, axis=1)

    date_cols = ['coldt', 'received_on:']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])

    df = concat_cols(df, 'collection', collection)

    return df


def get_visit_order(lims):
    visit_order = (lims[['subject', 'visit', 'coldt']].drop_duplicates()
                   .sort_values(['subject', 'coldt']).copy())
    visit_order = visit_order.drop_duplicates(['subject', 'visit'],
                                              keep='last')

    visit_order['count'] = visit_order.groupby(['subject']).cumcount()

    visit_order = (visit_order[['visit', 'count']].groupby('visit').mean()
                   .sort_values("count").round(1)
                   .rename({"count": 'order'}, axis=1).astype(str))
    visit_order['order'] = visit_order['order'].str.zfill(5)
    display(visit_order)

    lims = lims.merge(visit_order, how='left', on='visit')
    lims['ordered_visit'] = lims.order + "_" + lims.visit
    return lims


##############################
# MyAssay
##############################

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

    df = df.loc[df.result != "Not Analyzed"]
    
    numeric_cols = ['dil_factor', 'mean_final_conc', 'sd', 'cv',
                    'lloq', 'uloq', 'reportable_range_low',
                    'reportable_range_high', 'hemoglobin']
    
    df['limsid'] = df.limsid.str.strip()
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='ignore')
    df['result_numeric'] = pd.to_numeric(df.result, errors='coerce')

    df['result_filled'] = df.result
    df['is_filled'] = False
    df.loc[df.result.isin(['< LLOQ', '> ULOQ', 'Invalid']), "is_filled"] = True 
    df.loc[df.result.isin(['Invalid', '> ULOQ']),
           'result_filled'] = df.mean_final_conc
    df.loc[df.result_filled == "< LLOQ",
           'result_filled'] = df.reportable_range_low
    df['result_filled'] = pd.to_numeric(df.result_filled, errors='coerce')

    df['run_timestamp'] = df['run_timestamp'].str.replace(
                          r"(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}) (?:.{1,10})",
                          r'\1', regex=True)
    df['run_timestamp'] = pd.to_datetime(df['run_timestamp'])
    
    return df


def mad_add_loqs(df, g, uloq=True, lloq=True, log=True, rotate=0):
    loqs = df.groupby('assay').agg({'reportable_range_high': 'min',
                                    'reportable_range_low': 'max'})
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
                if loqs.shape[0] != 1:
                    uloq_level = loqs.loc[re.search(r'=\s(.*)$',
                                          ax.get_title()).group(1),
                                          'reportable_range_high']
                else:
                    uloq_level = loqs.iloc[0]['reportable_range_high']

                ax.axhline(uloq_level, ls='--', c='black')

            if lloq:
                if loqs.shape[0] != 1:
                    lloq_level = loqs.loc[re.search(r'=\s(.*)$',
                                          ax.get_title()).group(1),
                                          'reportable_range_low']
                else:
                    lloq_level = loqs.iloc[0]['reportable_range_low']

                ax.axhline(lloq_level, ls='--', c='black')


##############################
# Olink
##############################

def read_olink(path, npx=True, delimiter=","):
    '''
    Read and pre-process an Olink file into a pandas Dataframe

    Args:
        path: Filepath to the Olink file
    '''

    df = pd.read_csv(path, delimiter=delimiter, dtype=object)
    df = clean_colnames(df)

    df['missingfreq'] = df.missingfreq.str.replace('%', '')
    df['plateid'] = df.plateid.str.upper().str.replace(r'_RUN', '')

    if npx:
        df['result'] = df.npx
    else:
        df['result'] = df.quantified_value

    numeric_cols = ["result", "missingfreq", "platelod", 
                    "qc_deviation_inc_ctrl", "qc_deviation_det_ctrl"]

    if npx:
        numeric_cols = numeric_cols + ['maxlod']
    else:
        numeric_cols = numeric_cols + ["platelql", "lloq", "uloq", "unit"]

    numeric_cols = [x for x in numeric_cols if x in df.columns.to_list()]

    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df['missingfreq'] = df.missingfreq / 100

    df.loc[:, "type"] = "sample"
    df.loc[df.sampleid.str.contains(r"^Neg"), "type"] = "negative"
    df.loc[df.sampleid.str.contains(r"^CA.*"), "type"] = "calibrator"
    df.loc[df.sampleid.str.contains(r"^IPC.*"), "type"] = "ipc"
    df.loc[df.sampleid.str.contains(r"^CS.*"), "type"] = "control"

    if npx:
        if 'maxlod' in df.columns.to_list():
            df['blq'] = df.result < df.maxlod
        else:
            df['blq'] = df.result < df.platelod
        df['out_of_range'] = df.blq
    else:
        df['blq'] = df.result < df.platelql
        df['alq'] = df.result > df.uloq
        df['out_of_range'] = (df.blq | df.alq)

    return df


def olink_add_loqs(df, g, uloq=True, lloq=True, log=True, rotate=0, npx=True):
    if npx:
        loqs = df.groupby('assay').agg({'maxlod': 'max'})
    else:
        loqs = df.groupby('assay').agg({'platelql': 'max', 'uloq': 'min'})

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
                lloq_level = loqs.loc[re.search(r'=\s(.*)$',
                                      ax.get_title()).group(1), 'maxlod']
                ax.axhline(lloq_level, ls='--', c='black')
    else:
        for i, ax in enumerate(axes):
            if uloq:
                uloq_level = loqs.loc[re.search(r'=\s(.*)$',
                                      ax.get_title()).group(1), 'uloq']
                ax.axhline(uloq_level, ls='--', c='black')

            if lloq:
                lloq_level = loqs.loc[re.search(r'=\s(.*)$',
                                      ax.get_title()).group(1), 'platelql']
                ax.axhline(lloq_level, ls='--', c='black')


def view_plate(df, report_column, well='well', run=None,
               plate=None, export=False):

    sub = df.copy()

    if not ((run is None) and (plate is None)):
        sub = sub.loc[(sub.run == run) & (sub.plate == plate)]

    if sub[well].str.contains(r"\|").any():
        column_order = ['1/2', '3/4', '5/6', '7/8', '9/10', '11/12']
        sub[['well1', 'well2']] = sub[well].str.split("|", expand=True)
        sub['row_check'] = sub.well1.str[0] == sub.well2.str[0]
    
        if not sub.row_check.all():
            display(sub)
            raise VallueError("Replicates are not contained to the same well")
    
        sub['rep_row'] = sub.well1.str[0]
        sub['rep_columns'] = sub.well1.str[1:] + "/" + sub.well2.str[1:]

        test_column_subset = set(sub.rep_columns.drop_duplicates()
                                 .to_list()).issubset(set(column_order))

        if not test_column_subset:
            display(sub.rep_columns.drop_duplicates().to_list())
            raise VallueError("Replicates are not contained to proper columns")
    else:
        column_order = ["1", "2", "3", "4", "5", "6", "7",
                        "8", "9", "10", "11", "12"]
        sub['rep_row'] = sub[well].str[0]
        sub['rep_columns'] = sub[well].str[1:]

    sub = sub.pivot(columns='rep_columns', index='rep_row',
                    values=report_column)
    
    new_columns = [col for col in column_order 
                   if col not in sub.columns.to_list()]
    
    for col in new_columns:
        sub[col] = np.NaN
        
    sub = sub[column_order]
    
    sub.index.name = report_column
    if not ((run is None) and (plate is None)):
        sub.columns.name = f'{run}, plate {plate}'
    
    if export:
        xv(sub)
        
    return sub

'''
Tools for working with scientific/clinical data
'''

import re
import os
import pandas as pd
import numpy as np
from IPython.display import display
import mzl
from pandas import DataFrame
from matplotlib.axes import Axes
from seaborn.axisgrid import FacetGrid
from typing import Optional, List


##############################
# LIMS
##############################

def read_lims(path: str,
              sheet_name: Optional[str] = None,
              subject_col: str = 'patient_id',
              collection: List[str] = ['subject', 'visit']) -> DataFrame:
    """
    Load and process a LIMS export file into a standardized pandas DataFrame.

    Args:
        path (str): Filepath to the LIMS exporty (Excel) file.
        sheet_name (Optional[str]): Name of the sheet to load. Loads the first 
                                    sheet if None.
        subject_col (str): Column name to treat as subject identifier.
        collection (List[str]): Columns to concatenate into a composite
                                collection identifier.

    Returns:
        pd.DataFrame: Preprocessed LIMS data with standardized column 
                      names and date formatting.
    """
    if sheet_name is not None:
        df = pd.read_excel(path, sheet_name=sheet_name, dtype=object)
    else:
        df = pd.read_excel(path, dtype=object)

    df = df.mzl.clean_colnames()
    rename_cols_dict = {'sample_#': 'limsid', 'project_#': 'project',
                        'sample_id': 'barcode', subject_col: 'subject',
                        'visit_code': 'visit', 
                        'sample_collection_timestamp': 'coldt'}
    
    df = df.rename(rename_cols_dict, axis=1)

    date_cols = ['coldt', 'received_on:']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])

    df = df.mzl.concat_cols('collection', collection)
    return df


def add_visit_order(lims: DataFrame,
                    preview: bool = False) -> DataFrame:
    """
    Imply a visit order based on collection timestamps from multiple subjects. 

    Args:
        lims (pd.DataFrame): Preprocessed LIMS dataframe.

    Returns:
        pd.DataFrame: LIMS dataframe with an additional `ordered_visit` column.
    """
    visit_order = (lims[['subject', 'visit', 'coldt']]
                   .drop_duplicates()
                   .sort_values(['subject', 'coldt'])
                   .copy())
    visit_order = visit_order.drop_duplicates(['subject', 'visit'],
                                              keep='last')
    visit_order['count'] = visit_order.groupby(['subject']).cumcount()

    visit_order = (visit_order[['visit', 'count']]
                   .groupby('visit')
                   .mean()
                   .sort_values("count")
                   .round(1)
                   .rename({"count": 'order'}, axis=1)
                   .astype(str))
    visit_order['order'] = visit_order['order'].str.zfill(5)
    if preview:
        display(visit_order)

    lims = lims.merge(visit_order, how='left', on='visit')
    lims['ordered_visit'] = lims.order + "_" + lims.visit
    return lims


##############################
# MyAssay
##############################

def read_mad(path: str) -> DataFrame:
    """
    Load and preprocess a MyAssay Desktop (MAD) result file into a 
    pandas DataFrame.

    Args:
        path (str): File path to the MAD data file.

    Returns:
        pd.DataFrame: Cleaned and formatted MAD data including numeric
        conversion and annotations.
    """
    df = pd.read_csv(path, delimiter='\t', dtype=object)
    df = df.mzl.clean_colnames()
    df = df.rename({
        'sample': 'limsid', '%cv': 'cv',
        'hemoglobin_mg/dl': 'hemoglobin',
        'reported_value': 'result'
    }, axis=1)

    df = df.loc[df.result != "Not Analyzed"]

    numeric_cols = ['dil_factor', 'mean_final_conc', 'sd', 'cv',
                    'lloq', 'uloq', 'reportable_range_low',
                    'reportable_range_high', 'hemoglobin']

    df['limsid'] = df.limsid.str.strip()
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='ignore')
    df['result_numeric'] = pd.to_numeric(df.result, errors='coerce')

    df['result_filled'] = df.result
    df['is_filled'] = False
    df.loc[df.result.isin(['< LLOQ', '> ULOQ', 'Invalid']), 'is_filled'] = True
    df.loc[df.result.isin(['Invalid', '> ULOQ']),
           'result_filled'] = df.mean_final_conc
    df.loc[df.result_filled == "< LLOQ",
           'result_filled'] = df.reportable_range_low
    df['result_filled'] = pd.to_numeric(df['result_filled'], errors='coerce')

    df['run_timestamp'] = df['run_timestamp'].str.replace(
        r"(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}) (?:.{1,10})",
        r'\1', regex=True)
    df['run_timestamp'] = pd.to_datetime(df['run_timestamp'])

    try:
        df['run'] = df.runid.str.split("_", expand=True)[0]
    except:
        df['run'] = np.NaN
     
    try:
        df['panel'] = df.runid.str.split("_", expand=True)[1]
    except:
        df['panel'] = np.NaN
        
    try:
        df['plate'] = df.runid.str.split("_", expand=True)[2]
    except:
        df['plate'] = np.NaN

    return df


def mad_add_loqs(df: DataFrame, g: FacetGrid,
                 uloq: bool = True, lloq: bool = True,
                 log: bool = True, rotate: int = 0) -> None:
    """
    Overlay assay limits of quantification (LLOQ and ULOQ) on MAD plots.

    Args:
        df (pd.DataFrame): Assay results including reportable ranges.
        g (seaborn.FacetGrid): Seaborn grid object containing the plots.
        uloq (bool): Whether to draw ULOQ lines.
        lloq (bool): Whether to draw LLOQ lines.
        log (bool): Whether to apply log scale on y-axis.
        rotate (int): Angle for x-axis labels rotation.

    Returns:
        None
    """
    loqs = df.groupby('assay').agg({
        'reportable_range_high': 'min',
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

def read_olink(path: str,
               npx: bool = True,
               delimiter: str = ",") -> DataFrame:
    """
    Load and process an Olink result CSV file into a pandas DataFrame.

    Args:
        path (str): File path to the Olink CSV export file.
        npx (bool): Whether to use NPX values
                    (if False, use quantified values).
        delimiter (str): Delimiter used in the CSV file.

    Returns:
        pd.DataFrame: Standardized Olink assay results.
    """
    df = pd.read_csv(path, delimiter=delimiter, dtype=object)
    df = df.mzl.clean_colnames()

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


def olink_add_loqs(df: DataFrame, g: FacetGrid,
                   uloq: bool = True, lloq: bool = True,
                   log: bool = True, rotate: int = 0,
                   npx: bool = True) -> None:
    """
    Args:
        df (pd.DataFrame): Olink data frame containing assay information.
        g (seaborn.FacetGrid): Grid of seaborn plots.
        uloq (bool): Display ULOQ line.
        lloq (bool): Display LLOQ (or LOD for NPX) line.
        log (bool): Use logarithmic scale for y-axis.
        rotate (int): Rotate x-axis labels.
        npx (bool): Indicates if NPX values are used.

    Returns:
        None
    """
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


##############################
# ELISA
##############################

def read_elisa(filepath):
    df = pd.read_excel(filepath, header=None, usecols="B:M", 
                       skiprows=10, nrows=24, dtype=str)
    
    filename = os.path.basename(filepath)

    run_match = re.search(r'([A-Z]{3}\d{2,5})', filename)
    run = run_match.group(1) if run_match else None

    plate_match = re.search(r'plate\s*(\d+)', filename, re.IGNORECASE)
    plate = plate_match.group(1) if plate_match else '1'

    row_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    col_labels = list(range(1, 13))

    records = []

    for i, row_letter in enumerate(row_labels):
        row_start = i * 3
        sample_names = df.iloc[row_start]
        descriptions = df.iloc[row_start + 1]
        values = df.iloc[row_start + 2]

        for j, col_number in enumerate(col_labels):
            well = f"{row_letter}{col_number}"
            sample = sample_names.iloc[j]
            description = descriptions.iloc[j]
            value = values.iloc[j]

            records.append({
                "file": filename,
                "run": run,
                "plate": plate,
                "well": well,
                "pos": sample,
                "name": description,
                "result": value
            })

    stacked_df = pd.DataFrame(records)
    
    stacked_df["result"] = pd.to_numeric(stacked_df["result"],
                                         errors="coerce")
    return stacked_df

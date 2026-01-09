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
from typing import Optional, List, Union


##############################
# LIMS
##############################

def read_lims(path: str,
              sheet_name: Optional[str] = None,
              subject_col: str = 'original_sample_id',
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

def read_mad(paths: Union[str, List[str]]) -> DataFrame:
    """
    Load and preprocess one or more MyAssay Desktop (MAD) result files 
    into a single pandas DataFrame.

    Args:
        paths (str or list of str): File path(s) to the MAD data file(s).

    Returns:
        pd.DataFrame: Cleaned and formatted MAD data including numeric
                      conversion and annotations.
    """
    if isinstance(paths, str):
        paths = [paths]  # Normalize to list

    all_dfs = []

    for path in paths:
        try:
            df = pd.read_csv(path, delimiter='\t', dtype=object)
        except:
            print(f"Error in processing the following file: {path}")
            raise
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
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric,
                                                  errors='ignore')
        
        df['is_reportable'] = True
        df.loc[df.result.isin(['> ULOQ', 'Invalid']), 'is_reportable'] = False

        df['result_numeric'] = pd.to_numeric(df.result, errors='coerce')

        df['result_filled'] = df.result
        df['is_filled'] = False
        df.loc[df.result.isin(['< LLOQ', '> ULOQ', 'Invalid']),
               'is_filled'] = True
        df.loc[df.result.isin(['Invalid', '> ULOQ']),
               'result_filled'] = df.mean_final_conc
        df.loc[df.result_filled == "< LLOQ",
               'result_filled'] = df.reportable_range_low
        df['result_filled'] = pd.to_numeric(df['result_filled'],
                                            errors='coerce')

        df['run_timestamp'] = df['run_timestamp'].str.replace(
            r"(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}) (?:.{1,10})",
            r'\1', regex=True)
        df['run_timestamp'] = pd.to_datetime(df['run_timestamp'])

        try:
            df['run'] = df.runid.str.split("_", expand=True)[0]
        except Exception:
            df['run'] = np.NaN
        try:
            df['panel'] = df.runid.str.split("_", expand=True)[1]
        except Exception:
            df['panel'] = np.NaN
        try:
            df['plate'] = df.runid.str.split("_", expand=True)[2]
        except Exception:
            df['plate'] = np.NaN

        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)


def read_mad_qc(paths: Union[str, List[str]]) -> DataFrame:
    """
    Load and preprocess one or more MyAssay Desktop (MAD) QC files 
    into a single pandas DataFrame.

    Args:
        paths (str or list of str): File path(s) to the MAD QC data file(s).

    Returns:
        pd.DataFrame: Cleaned and formatted MAD QC data including numeric
                      conversion and annotations.
    """
    if isinstance(paths, str):
        paths = [paths]  # Normalize to list

    all_dfs = []

    for path in paths:
        df = pd.read_csv(path, delimiter='\t', dtype=object)
        df = df.mzl.clean_colnames()

        df = df.rename({"assayname": "panel", 
                        "analyte": 'assay',
                        "runname": "runid",
                        "rundate": "run_timestamp",
                        "barcode": "msd_barcode"}, axis=1)

        df['run_timestamp'] = pd.to_datetime(df.run_timestamp, errors='coerce')
        df['nominalconc'] = pd.to_numeric(df.nominalconc, errors='coerce')
        df['finalconc'] = pd.to_numeric(df.finalconc, errors='coerce')
        df['bias'] = pd.to_numeric(df.bias, errors='coerce')

        all_dfs.append(df)

    all_dfs = pd.concat(all_dfs, ignore_index=True)
    all_dfs = all_dfs.sort_values('run_timestamp')
    return all_dfs


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

def read_olink(paths: Union[str, List[str]],
               delimiter: str = ";") -> DataFrame:
    """
    Load and process an Olink result CSV file into a pandas DataFrame.

    Args:
        paths (str or list of str): File path(s) to the Olink CSV
                                    export file(s).
        delimiter (str): Delimiter used in the CSV file.

    Returns:
        pd.DataFrame: Standardized Olink assay results.
    """
    if isinstance(paths, str):
        paths = [paths]

    all_dfs = []

    for path in paths:
        df = pd.read_csv(path, delimiter=delimiter, dtype=object)
        df = df.mzl.clean_colnames()

        if "olink_npx_signature_version" in df.columns:
            report_software_version = 1
            df = df.rename({'panel_version': 'panelversion',
                            'quantified_value': 'quantifiedvalue',
                            'platelod': 'lodquant',
                            'platelql': 'lql',
                            'olink_npx_signature_version': 'softwareversion',
                            'qc_deviation_inc_ctrl': 'qcdeviationdetctrl',
                            'qc_deviation_det_ctrl': 'qcdeviationincctrl',
                            'assay_warning': 'assayqc',
                            'qc_warning': 'sampleqc'},
                           axis=1)

            # removed content columns:
            # Index, QC_WarningPlateLQL, MaxLOD

        else:
            report_software_version = 2
            # new content columns:
            # Product, WellID, SampleType, Ct, LODNPX, BelowLOD, 
            # BelowLQL, AboveULOQ

            # LQL is LLOQ if plate LOD is lower, or plate LOD otherwise 

        df['missingfreq'] = df.missingfreq.str.replace('%', '')
        df['plateid'] = (df.plateid.str.upper()
                         .str.replace(r'([A-Z])_RUN(\d{2,4}).*', r'\1\2', 
                         regex=True))

        npx = not 'quantifiedvalue' in df.columns

        if npx:
            df['result'] = df.npx
        else:
            df['result'] = df.quantifiedvalue

        numeric_cols = ["result", "missingfreq", "ct", 'maxlod',
                        "npx", "lodnpx", "lodquant", "lloq", "lql",
                        "uloq", "qcdeviationdetctrl", "qcdeviationincctrl"]

        numeric_cols = [x for x in numeric_cols if x in df.columns.to_list()]

        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric,
                                                  errors='coerce')

        boolean_cols = ['belowlod']

        boolean_cols = [x for x in boolean_cols if x in df.columns.to_list()]

        for col in boolean_cols:
            df[col] = df[col].map({"TRUE": True, 'FALSE': False}).astype(bool)

        if report_software_version == 1:
            df['missingfreq'] = df.missingfreq / 100

            df.loc[:, "sampletype"] = "SAMPLE"
            df.loc[df.sampleid.str.contains(r"^Neg"),
                   "sampletype"] = "NEGATIVE_CONTROL"
            df.loc[df.sampleid.str.contains(r"^CA.*"),
                   "sampletype"] = "CALIBRATOR"
            df.loc[df.sampleid.str.contains(r"^IPC.*"),
                   "sampletype"] = "IPC"
            df.loc[df.sampleid.str.contains(r"^CS.*"),
                   "sampletype"] = "CONTROL"

        if npx:
            df['blq'] = df.result < df.lodnpx
            df['out_of_range'] = df.blq
        else:
            df['blq'] = df.result < df.lql
            df['alq'] = df.result > df.uloq
            df['out_of_range'] = (df.blq | df.alq)

        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)


def olink_add_loqs(df: DataFrame, g: FacetGrid,
                   uloq: bool = True, lloq: bool = True,
                   log: bool = True, rotate: int = 0) -> None:
    """
    Args:
        df (pd.DataFrame): Olink data frame containing assay information.
        g (seaborn.FacetGrid): Grid of seaborn plots.
        uloq (bool): Display ULOQ line.
        lloq (bool): Display LLOQ (or LOD for NPX) line.
        log (bool): Use logarithmic scale for y-axis.
        rotate (int): Rotate x-axis labels.

    Returns:
        None
    """

    npx = 'quantifiedvalue' in df.columns 

    if npx:
        loqs = df.groupby('assay').agg({'maxlod': 'max'})
    else:
        loqs = df.groupby('assay').agg({'lql': 'max', 'uloq': 'min'})
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
                                      ax.get_title()).group(1), 'lql']
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

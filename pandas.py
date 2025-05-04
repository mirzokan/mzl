'''
Tools for Pandas
'''

import os
import platform
import subprocess
import re
import tempfile
import glob
import time
from datetime import datetime as dt

import pandas as pd
import numpy as np
import pandas.io.sql as psql

from IPython.display import display

import psycopg2
from configparser import ConfigParser
from mzl import subl


def read_config(filename='db.ini', section='postgresql'):
    '''
    Reads an .ini configuration file meant to hold database connection
    configuration

    Arguments:
    filename: String, path to the configuraiton file
    section: String, name of the .ini file section to read

    Returns: (Dictionary) containing configuration settings
    '''
    parser = ConfigParser()
    parser.read(filename)

    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception(f'Section {section} not found in the {filename} file')
    return db


def db_reader(filename='db.ini', section='postgresql'):
    '''
    Creates a database connection object and a function to read SQL 
    straight into a pandas DataFrame

    Arguments:
    filename: String, path to the configuraiton file
    section: String, name of the .ini file section to read

    Returns: (Tuple) first a database connection object, second, a 
    function that takes a string SQL query and returns the database 
    response as a pandas DataFrame.
    '''
    db = read_config(filename=filename, section=section)
    conn = psycopg2.connect(**db)

    def dr(sql):
        return psql.read_sql(sql, conn)

    return conn, dr


@pd.api.extensions.register_dataframe_accessor("mzl")
class MzlAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def xv(self, index=True, label=''):
        """
        Save a DataFrame as a temporary Excel file and open it.
        Non-destructive: does not modify the original DataFrame.
        """
        oldfiles = glob.glob(os.path.join(tempfile.gettempdir(), "mzl_xview_*"))

        try:
            for file in oldfiles:
                time_alive = time.time() - os.path.getctime(file)
                if time_alive > 60:
                    os.remove(os.path.abspath(file))
        except:
            pass

        if label != '':
            prefix = f"mzl_xview_{label}_{dt.now().strftime('%Y-%m-%d_%H-%M')}_"
        else:
            prefix = f"mzl_xview_{dt.now().strftime('%Y-%m-%d_%H-%M')}_"

        tf = tempfile.NamedTemporaryFile(prefix=prefix, suffix=".xlsx",
                                         delete=False)
        self._obj.to_excel(tf, index=index)
        path = os.path.abspath(tf.name)
        tf.close()
        
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.call(["open", path])
        else:
            subprocess.call(["xdg-open", path])

    def push_cols(self, pushcols, back=False):
        """
        Reorder columns by pushing selected ones to front or back.
        Returns a modified copy.
        """
        df = self._obj.copy()
        if back:
            return df[subl(df.columns, pushcols) + pushcols]
        else:
            return df[pushcols + subl(df.columns, pushcols)]

    def merge_duplicate_rows(self, groupby, delimiter="|"):
        """
        Deduplicate rows by merging non-groupby column values.
        Returns a modified copy.
        """
        df = self._obj.copy()

        def merge_apply(group, groupby, delimiter, columns):
            merged_group = group.iloc[0].copy()
            for col in columns:
                col_values = list(dict.fromkeys(group[col]))
                col_values = [str(x) for x in col_values if str(x).lower() not in ['', 'nan', '-', 'n/ap']]
                merged_group[col] = delimiter.join(col_values)
            return merged_group

        if not isinstance(groupby, list):
            groupby = [groupby]

        columns = subl(df.columns.tolist(), groupby)
        unique = df.loc[~df.duplicated(subset=groupby, keep=False)].copy()
        duplicated = df.loc[df.duplicated(subset=groupby, keep=False)].copy()

        duplicated = (duplicated.groupby(groupby, as_index=False, group_keys=False)
                      .apply(merge_apply, groupby=groupby, delimiter=delimiter, columns=columns))

        result = (pd.concat([unique, duplicated])
                  .sort_values(groupby)
                  .reset_index(drop=True))

        return result

    def concat_cols(self, colname, collist, joinstring="_"):
        """
        Concatenate multiple columns into one new column.
        Returns a modified copy.
        """
        df = self._obj.copy()
        ddf = df[collist].astype(str)
        df[colname] = ddf.apply(lambda x: joinstring.join(x), axis=1)
        return df

    def clean_colnames(self):
        """
        Sanitize column names for consistency. Returns a modified copy.
        """
        df = self._obj.copy()
        df.columns = (df.columns.str.lower()
                      .str.replace(r"\s|-", "_", regex=True)
                      .str.replace(r"\\.", "", regex=True)
                      .str.replace(r"[\(\)<>\?]", "", regex=True)
                      .str.replace(r"_{2,}", "_", regex=True)
                      .str.replace(r"(^_+|_+$)", "", regex=True))
        return df

    def pretty_colnames(self, renames=None):
        """
        Format column names for display: capitalize and clean underscores.
        Returns a modified copy.
        """
        def capital_one(key):
            return re.sub(r'^([a-zA-Z])', lambda x: x.groups()[0].upper(), key)

        df = self._obj.copy()
        if renames is not None:
            df = df.rename(columns=renames)
        df.columns = df.columns.str.replace("_", " ", regex=False)
        df.columns = [capital_one(col) for col in df.columns]
        return df

    def view_plate(self, report_column, well='well', run=None,
                   plate=None, export=False):
        sub = self._obj.copy()

        if not ((run is None) and (plate is None)):
            sub = sub.loc[(sub.run == run) & (sub.plate == plate)]

        if sub[well].str.contains(r"\|").any():
            column_order = ['1/2', '3/4', '5/6', '7/8', '9/10', '11/12']
            sub[['well1', 'well2']] = sub[well].str.split("|", expand=True)
            sub['row_check'] = sub.well1.str[0] == sub.well2.str[0]

            if not sub.row_check.all():
                display(sub)
                raise ValueError("Replicates are not contained to the same well")

            sub['rep_row'] = sub.well1.str[0]
            sub['rep_columns'] = sub.well1.str[1:] + "/" + sub.well2.str[1:]

            test_column_subset = set(sub.rep_columns.drop_duplicates()
                                     .to_list()).issubset(set(column_order))

            if not test_column_subset:
                display(sub.rep_columns.drop_duplicates().to_list())
                raise ValueError("Replicates are not contained to proper columns")
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
            sub.mzl.xv()  # assumes you have an 'xv' method registered similarly

        return sub

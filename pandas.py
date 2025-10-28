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
from typing import Optional, Callable, Tuple, Union, List, Dict


def read_config(filename: str = 'db.ini',
                section: str = 'postgresql') -> Dict[str, str]:
    """
    Reads database connection parameters from a .ini configuration file.

    Args:
        filename (str): Path to the configuration file. Defaults to 'db.ini'.
        section (str): Section name in the .ini file to read. 
                       Defaults to 'postgresql'.

    Returns:
        Dict[str, str]: Dictionary containing configuration key-value pairs.

    Raises:
        Exception: If the section is not found in the configuration file.
    """
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


def db_reader(filename: str = 'db.ini',
              section: str = 'postgresql'
              ) -> Tuple[psycopg2.extensions.connection,
                         Callable[[str], pd.DataFrame]]:
    """
    Creates a PostgreSQL database connection and a reader function.

    Args:
        filename (str): Path to the configuration file. Defaults to 'db.ini'.
        section (str): Section name to read connection parameters from. 
                       Defaults to 'postgresql'.

    Returns:
        Tuple[psycopg2.extensions.connection, Callable[[str], pd.DataFrame]]: 
            A connection object and a function to execute SQL queries 
            into DataFrames.
    """
    db = read_config(filename=filename, section=section)
    conn = psycopg2.connect(**db)

    def dr(sql):
        return psql.read_sql(sql, conn)

    return conn, dr


@pd.api.extensions.register_dataframe_accessor("mzl")
class MzlAccessor:
    """
    A pandas DataFrame accessor for common preprocessing, formatting, 
    and exploratory tools. Registered under the `.mzl` namespace.
    """

    def __init__(self, pandas_obj: pd.DataFrame) -> None:
        self._obj = pandas_obj

    def xv(self, index: bool = False, label: str = '') -> None:
        """
        Save the DataFrame to a temporary Excel file and open it. Used as
        a lazy data viewer.

        Args:
            index (bool): Whether to include the index in the Excel file.
                          Defaults to True.
            label (str): Optional label for naming the temporary file.
        """
        oldfiles = glob.glob(os.path.join(tempfile.gettempdir(),
                                          "mzl_xview_*"))

        try:
            for file in oldfiles:
                time_alive = time.time() - os.path.getctime(file)
                if time_alive > 60:
                    os.remove(os.path.abspath(file))
        except:
            pass

        if label != '':
            prefix = (f"mzl_xview_{label}_"
                      f"{dt.now().strftime('%Y-%m-%d_%H-%M')}_")
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

    def push_cols(self, 
                  pushcols: List[str],
                  back: bool = False) -> pd.DataFrame:
        """
        Move selected columns to the front or back of the DataFrame.

        Args:
            pushcols (List[str]): Columns to reposition.
            back (bool): If True, push to the end instead of the front.

        Returns:
            pd.DataFrame: Reordered DataFrame.
        """
        df = self._obj.copy()
        if back:
            return df[subl(df.columns, pushcols) + pushcols]
        else:
            return df[pushcols + subl(df.columns, pushcols)]

    def merge_duplicate_rows(self, 
                             groupby: Union[str, List[str]],
                             delimiter: str = "|") -> pd.DataFrame:
        """
        Merge rows with duplicate keys by concatenating other column values.

        Args:
            groupby (Union[str, List[str]]): Column(s) to group by.
            delimiter (str): Delimiter used to join values. Defaults to '|'.

        Returns:
            pd.DataFrame: DataFrame with deduplicated and merged rows.
        """
        df = self._obj.copy()

        def merge_apply(group: pd.DataFrame, groupby: List[str],
                        delimiter: str, columns: List[str]) -> pd.Series:
            merged_group = group.iloc[0].copy()
            for col in columns:
                col_values = list(dict.fromkeys(group[col]))
                col_values = [str(x) for x in col_values if str(x).lower() 
                              not in ['', 'nan', '-', 'n/ap']]
                merged_group[col] = delimiter.join(col_values)
            return merged_group

        if not isinstance(groupby, list):
            groupby = [groupby]

        columns = subl(df.columns.tolist(), groupby)
        unique = df.loc[~df.duplicated(subset=groupby, keep=False)].copy()
        duplicated = df.loc[df.duplicated(subset=groupby, keep=False)].copy()

        duplicated = (duplicated.groupby(groupby, as_index=False,
                                         group_keys=False)
                      .apply(merge_apply, 
                             groupby=groupby,
                             delimiter=delimiter,
                             columns=columns))

        result = (pd.concat([unique, duplicated])
                  .sort_values(groupby)
                  .reset_index(drop=True))

        return result

    def concat_cols(self, 
                    colname: str, 
                    collist: List[str],
                    joinstring: str = "_") -> pd.DataFrame:
        """
        Concatenate multiple columns into a new column.

        Args:
            colname (str): Name of the new column.
            collist (List[str]): List of columns to concatenate.
            joinstring (str): Separator string. Defaults to '_'.

        Returns:
            pd.DataFrame: Modified DataFrame with new concatenated column.
        """
        df = self._obj.copy()
        ddf = df[collist].astype(str)
        df[colname] = ddf.apply(lambda x: joinstring.join(x), axis=1)
        return df

    def clean_colnames(self) -> pd.DataFrame:
        """
        Standardize and clean column names by removing special characters,
        normalizing case and spacing.

        Returns:
            pd.DataFrame: DataFrame with cleaned column names.
        """
        df = self._obj.copy()
        df.columns = (df.columns.str.lower()
                      .str.replace(r"\s|-", "_", regex=True)
                      .str.replace(r"\.", "", regex=True)
                      .str.replace(r"[\(\)<>\?\*]", "", regex=True)
                      .str.replace(r"_{2,}", "_", regex=True)
                      .str.replace(r"(^_+|_+$)", "", regex=True))
        return df

    def pretty_colnames(self, 
                        renames: Optional[Dict[str, str]] = None
                        ) -> pd.DataFrame:
        """
        Beautify column names for readability and for presentation.

        Args:
            renames (Optional[Dict[str, str]]): Optional dictionary to 
                                                rename columns.

        Returns:
            pd.DataFrame: DataFrame with formatted column names.
        """
        def capital_one(key: str) -> str:
            return re.sub(r'^([a-zA-Z])', lambda x: x.groups()[0].upper(), key)

        df = self._obj.copy()
        if renames is not None:
            df = df.rename(columns=renames)
        df.columns = df.columns.str.replace("_", " ", regex=False)
        df.columns = [capital_one(col) for col in df.columns]
        return df

    def view_plate(self,
                   report_column: str,
                   well: str = 'well',
                   run: Optional[str] = None,
                   plate: Optional[str] = None,
                   export: bool = False) -> pd.DataFrame:
        """
        Generate a plate-format view of well data, optionally filtering 
        by run/plate.

        Args:
            report_column (str): Column to use for cell values.
            well (str): Column that indicates the well identifier. 
                        Defaults to 'well'.
            run (Optional[str]): Optional filter for run.
            plate (Optional[str]): Optional filter for plate.
            export (bool): If True, exports the view to Excel.

        Returns:
            pd.DataFrame: Pivoted DataFrame representing plate layout.
        
        Raises:
            ValueError: If replicate wells are invalid or not aligned 
                        correctly.
        """
        sub = self._obj.copy()

        if not ((run is None) and (plate is None)):
            sub = sub.loc[(sub.run == run) & (sub.plate == plate)]

        if sub[well].str.contains(r"\|").any():
            column_order = ['1/2', '3/4', '5/6', '7/8', '9/10', '11/12']
            sub[['well1', 'well2']] = sub[well].str.split("|", expand=True)
            sub['row_check'] = sub.well1.str[0] == sub.well2.str[0]

            if not sub.row_check.all():
                display(sub)
                raise ValueError("Replicates are not contained "
                                 "to the same well")

            sub['rep_row'] = sub.well1.str[0]
            sub['rep_columns'] = sub.well1.str[1:] + "/" + sub.well2.str[1:]

            test_column_subset = set(sub.rep_columns.drop_duplicates()
                                     .to_list()).issubset(set(column_order))

            if not test_column_subset:
                display(sub.rep_columns.drop_duplicates().to_list())
                raise ValueError("Replicates are not contained "
                                 "to the same well")
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
            sub.mzl.xv()

        return sub

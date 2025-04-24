'''
Tools for Pandas
'''

import os
import re
import tempfile
import glob
import time
from datetime import datetime as dt
import string

import pandas as pd
import numpy as np
import pandas.io.sql as psql

from IPython.display import display, Markdown

import psycopg2
from configparser import ConfigParser
from .gen import subl


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


def xview(df, index=True, label=''):
    '''
    Save a Pandas DataFrame as a temporary Excel file and open it, to 
    be used as a lazy data viewer.

    Arguments:
    df: Pandas Dataframe to view
    '''
    
    oldfiles = glob.glob(os.path.join(tempfile.gettempdir(),
                                      "mzl_xview_*"))

    try:
        for file in oldfiles:
            time_alive = now = time.time() - os.path.getctime(file)
            if time_alive > 60:
                os.remove(os.path.abspath(file))
    except:
        pass

    if label != '':
        prefix = f"mzl_xview_{label}_{dt.now().strftime('%Y-%m-%d_%H-%M')}_"
    else:
        prefix = f"mzl_xview_{dt.now().strftime('%Y-%m-%d_%H-%M')}_"
    
    tf = tempfile.NamedTemporaryFile(prefix=prefix,
                                     suffix=".xlsx", delete=False)
    df.to_excel(tf, index=index)
    path = os.path.abspath(tf.name) 
    com = r"start {}".format(path)
    tf.close()
    os.system(com)


# Alias for xview
xv = xview


def push_cols(df, pushcols, back=False):
    """Pushes a list of columns to the front (or back) of a DataFrame
    
    Args:
        df (DataFrame): Dataframe to reorder columns
        pushcols (list): Ordered list of columns to push to front
        back (boolean): Pushes the columns to the back of the list when
                        set to True
    
    Returns:
        DataFrame: Dataframe with reordered columns
    """
    if back:
        return df[subl(df.columns, pushcols) + pushcols]
    else:
        return df[pushcols + subl(df.columns, pushcols)]


def merge_duplicate_rows(df, groupby, delimiter="|"):
    """Takes a dataframe and a list of ID columns, then deduplicates
    the dataframe such that the IDs become unique, while values in 
    non-ID columns are concatenated.

    Args:
        df (Dataframe): Dataframe with rows that have duplicate IDs
        groupby (str or list): Name of a column or list of columns
                                  to serve as row IDs.
        delimiter (str): Delimiter string to separate concatenated 
                         values.

    Returns:
        Dataframe: Dataframe where row IDs are deduplicated.
    """

    def merge_apply(group, groupby, delimiter, columns):
        merged_group = group.iloc[0].copy()

        for col in columns:
            col_values = list(dict.fromkeys(group[col]))
            col_values = [str(x) for x in col_values if str(x).lower() 
                          not in ['', 'nan', '-', 'n/ap']]
            col_values = delimiter.join(col_values)

            merged_group[col] = col_values

        return merged_group

    if type(groupby) != list:
        groupby = [groupby]

    columns = df.columns
    columns = subl(columns, groupby)

    unique = df.loc[~df.duplicated(subset=groupby, keep=False)].copy()
    duplicated = df.loc[df.duplicated(subset=groupby, keep=False)].copy()

    duplicated = (duplicated.groupby(groupby, 
                  as_index=False, group_keys=False)
                  .apply(merge_apply, groupby=groupby,
                  delimiter=delimiter, columns=columns))

    df = (pd.concat([unique, duplicated]).sort_values(groupby)
          .reset_index(drop=True))

    return df


def concat_cols(df, colname, collist, joinstring="_"):
    """Simple helper function to concatenate values of multiple
       columns into one new column.
    
    Args:
        df (TYPE): Dataframe on which to perform the concatenation
        colname (TYPE): Name of the new column containing the catenations
        collist (TYPE): List of columns to concatenate
        joinstring (str, optional): String to use in between the concatenated
                                    values
    
    Returns:
        Dataframe: Dataframe with the new column of concatenated values
    """
    ddf = df[collist].copy().astype(str)
    df[colname] = ddf.apply(lambda x:
                            joinstring.join(x),
                            axis=1)
    return df


def clean_colnames(df):
    """Cleans up column names in a DataFrame to make it more Pandas
       friendly. 
    
    Args:
        df (DataFrame): DataFrame with column names cleaned up
    """
    df.columns = (df.columns.str.lower()
                  .str.replace(r"\s|-", r"_", regex=True))
    df.columns = (df.columns.str.lower()
                  .str.replace(r"\.", r"", regex=True))
    df.columns = (df.columns.str.lower()
                  .str.replace(r"[\(|\)|<|>|\?]", r"", regex=True))
    df.columns = (df.columns.str.lower()
                  .str.replace(r"_{2,}", r"_", regex=True))
    df.columns = (df.columns.str.lower()
                  .str.replace(r"(?:^_+)|(?:_+$)", r"", regex=True))
    return df

def pretty_colnames(df, renames=None):
    """Makes column names in a DataFrame more amenable to presentation. 
    
    Args:
        df (DataFrame): DataFrame with column names prettified
        renames (dictionary): Dictionary with additional renames
    """
    def capital_one(key):
        return re.sub('([a-zA-Z])', lambda x: x.groups()[0].upper(), key, 1)

    sub = df.copy()

    if not renames is None:
        sub = sub.rename(renames, axis=1)

    sub.columns = sub.columns.str.replace("_", r" ", regex=False)
    columns_list = sub.columns.to_list()

    columns_list = [capital_one(col) for col in columns_list]

    sub.columns = columns_list

    return sub


def view_plate(df, report_column, well='well', run=None,
               plate=None, export=False):

    sub = df.copy()

    if not ((run is None) and (plate is None)):
        sub = sub.loc[(sub.run == run) & (sub.plate == plate)]

    if sub[well].str.contains("\|").any():
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
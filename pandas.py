'''
Tools for Pandas
'''

import os
import tempfile
import time
import glob
from datetime import datetime as dt

import pandas as pd
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
    """Pushes a list of columns to the front of the DataFrame
    
    Args:
        df (DataFrame): Dataframe to reorder columns
        pushcols (list): Ordered list of columns to push to front
        back (boolean): Pushes the columns to the back of the list when
                        set to True
    
    Returns:
        TYPE: DataFrame
    """
    if back:
        return df[subl(df.columns, pushcols) + pushcols]
    else:
        return df[pushcols + subl(df.columns, pushcols)]


def merge_duplicate_rows(df, groupby, delimiter="|"):

    def merge_apply(group, groupby, delimiter):
        if type(groupby) != list:
            groupby = list(groupby)

        columns = group.columns
        columns = subl(columns, groupby)

        merged_group = group.iloc[0].copy()

        for col in columns:
            col_values = group[col].drop_duplicates()
            col_values = [str(x) for x in col_values if str(x).lower() 
                          not in ['', 'nan', '-', 'n/ap']]
            col_values = delimiter.join(col_values)

            merged_group[col] = col_values

        return merged_group

    unique = df.loc[~df.duplicated(subset=groupby, keep=False)].copy()
    duplicated = df.loc[df.duplicated(subset=groupby, keep=False)].copy()

    duplicated = (duplicated.groupby(groupby, 
                  as_index=False, group_keys=False)
                  .apply(merge_apply, groupby=groupby,
                  delimiter=delimiter))

    df = (pd.concat([unique, duplicated]).sort_values(groupby)
          .reset_index(drop=True))

    return df

# def merge_duplicate_rows(group, delimiter="|", cols=None):
#     """Takes a dataframe grouped by an index-like columns that may 
#     contain duplicates. Joins duplicate entries by a pipe
    
#     Args:
#         group (Pandas Groupby): Description
    
#     Returns:
#         Pandas Groupby: Group with merged duplicates
#     """

#     if group.shape[0] == 1:
#         return group
    
#     merged_set = group.iloc[0]

#     if cols is None:
#         cols = group.columns

#     for col in cols:
#         colset = group[col].drop_duplicates()
#         if colset.shape[0] == 1:
#             try:
#                 merged_set.loc[col] = colset.iloc[0]
#             except:
#                 print(f"merged_set: {merged_set}")
#                 print(f"colset: {colset}")
#                 raise
#         else:
#             try:
#                 colset = [str(x) for x in colset if str(x).lower() 
#                           not in ['', 'nan', '-', 'n/ap']]
#                 if len(colset) == 1:
#                     merged_set.loc[col] = colset
#                 else:
#                     merged_set.loc[col] = delimiter.join(colset)
#             except:
#                 display(group)
#                 raise
#                 # return group
            
#     return merged_set


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

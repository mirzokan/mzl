'''
Tools for Pandas
'''

import re, sys, os, itertools, tempfile, glob

import numpy as np
import pandas as pd
import pandas.io.sql as psql

import psycopg2
from configparser import ConfigParser


def read_config(filename='db.ini', section='postgresql'):
    '''
    Reads an .ini configuration file meant to hold database connection configuration

    Arguments:
    filename: String, path to the configuraiton file
    section: String, name of the .ini file section to read

    Returns:
    A dictionary of configuration settings
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
    Creates a database connection object and a function to read SQL straight into a pandas DataFrame

    Arguments:
    filename: String, path to the configuraiton file
    section: String, name of the .ini file section to read

    Returns:
    A tuple, first a database connection object, secornd, a function that takes a string 
    SQL query and returns the database response as a pandas DataFrame
    '''
    db = read_config(filename=filename, section=section)
    conn = psycopg2.connect(**db)

    def dr(sql):
        psql.read_sql(sql, conn)

    return conn, dr


def xview(df):
    '''
    Save a Pandas DataFrame as an temporary Excel file and open it, to be used as a lazy data viewer

    Arguments:
    df: Pandas Dataframe to view
    '''
    
    oldfiles = glob.glob(os.path.join(tempfile.gettempdir(), "mizosoup_xview_*"))

    try:
        for file in oldfiles:
            os.remove(os.path.abspath(file))
    except:
        pass
    
    tf = tempfile.NamedTemporaryFile(prefix="mizosoup_xview_", suffix=".xlsx", delete=False)
    df.to_excel(tf)
    path = os.path.abspath(tf.name) 
    com = r"start {}".format(path)
    tf.close()
    os.system(com)

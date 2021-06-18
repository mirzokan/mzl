'''
Tools for Pandas
'''

import re, sys, os, itertools

import numpy as np
import pandas as pd
import pandas.io.sql as psql

import psycopg2
from configparser import ConfigParser


# def config(filename='db.ini', section='postgresql'):
#     parser = ConfigParser()
#     parser.read(filename)

#     db = {}
#     if parser.has_section(section):
#         params = parser.items(section)
#         for param in params:
#             db[param[0]] = param[1]
#     else:
#         raise Exception(f'Section {section} not found in the {filename} file')
#     return db

# def dr(sql):
#     global conn
#     return psql.read_sql(sql, conn)

def xview(df):
'''
Save a Pandas DataFrame as an temporary Excel file and open it, to be used as a lazy data viewer
'''
    current_path = os.getcwd()
    filename = "view.xlsx"
    try:
        df.to_excel(filename)
    except:
        filename = "view2.xlsx"
        df.to_excel(filename)
    full_path = os.path.join(current_path, filename)
    os.system("start " + full_path)
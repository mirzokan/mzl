'''
General Purpose Tools
'''

import re
import os
import shutil
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy


class CustomException(Exception):
    pass


def sdir(obj, sunder=False):
    '''
    Modification of the dir function to detail special and object
    specific callables and attributes. Prints out Callables and 
    Attributes.

    Args:
        obj: An object to instpect
        sunder: Boolean, False shows non-undered attributes,
                True includes single-under attributes
    '''

    print(type(obj))
    if sunder:
        unders = [x for x in dir(obj) if re.match("^_(?!_)", x)]
        callables = [x for x in unders if callable(getattr(obj, x))]
        attribs = [x for x in unders if x not in callables]
        print("--Special--")
        print("Callables:")
        print(callables)
        print("Attributes:")
        print(attribs)
    
    regs = [x for x in dir(obj) if re.match("^[^_]", x)]
    callables = [x for x in regs if callable(getattr(obj, x))]
    attribs = [x for x in regs if x not in callables]
    print("")
    print("--Regular--")
    print("Callables:")
    print(callables)
    print("Attributes:")
    print(attribs)


def subl(a, b):
    """
    Subtract list b from list a.
    
    Args:
        a (list): List from which to subtract
        b (list): List which to subtract
    
    Returns:
        list: List that results from the subtraction
    """
    return [x for x in a if x not in b]


def get_latest_file(path, pattern=None):
    """
    From a specific folder, get a list of files, optionally matching a
    regex pattern, and return the last file in alphabetical order.
    Useful for getting the latest file from a series that contain
    timestamps in the filename.

    
    Args:
        path (str): Path to the folder to scan.
        pattern (str, optional): Optional regex pattern to use as
        a filter
    
    Returns:
        TYPE: str
    
    Raises:
        FileNotFoundError: When no matching files are found.
    """
    files = next(os.walk(path))[2]
    if pattern is not None:
        files = [file for file in files 
                 if re.fullmatch(pattern, file) is not None]
    files.sort(reverse=True)
    if len(files) >= 1:
        latest_file = os.path.join(path, files[0])
    else:
        raise FileNotFoundError
    return latest_file


def list_files_by_ext(path, ext, excluded_subfolder_terms=None):
    """
    From a specific root folder, get a list of files that have a 
    specified file extension in the subfolder structure, filtered by 
    exclusion terms.
    
    Args:
        path (str): Path to the folder to scan.
        ext (str): File extension to filter for.
        excluded_subfolder_terms (list): List of terms contained in the 
                                         folder paths to exclude
    
    Returns:
        TYPE: list
    """
    file_list = []
    
    for root, dirs, files in os.walk(path):
        if excluded_subfolder_terms is not None:
            excluded_subfolder_terms = [item.lower() 
                                        for item in excluded_subfolder_terms]
            exclusion_filters = [exclusion in root.lower()
                                 for exclusion in excluded_subfolder_terms]
            if any(exclusion_filters):
                continue
        for file in files:
            filepath = os.path.join(root, file)
            if ((os.path.splitext(filepath)[1] == ext) 
               and ("~" not in filepath)):
                file_list.append(filepath)
    return file_list


def copy_files(file_list, destination_path, message=False):
    """Copies a list of paths to a destination directory.
    Does not overwrite existing files.
    
    Args:
        file_list (list): List of paths to copy
        destination_path (str): Destination path
        message (str or bool, optional): Whether results are printed,
                set to "all" to print previously existing files.
    """
    for i, file_path in enumerate(file_list):
        file_name = os.path.basename(file_path)
        new_path = os.path.join(destination_path, file_name)
        if os.path.isfile(new_path):
            if message == "all":
                print(f"{i}. Exists: {os.path.basename(file_name)}")
        else:
            shutil.copy(file_path, new_path)
            if message is not False:
                print(f"{i}. Copied: {os.path.basename(file_name)}")


class execution_timer(object):
    """docstring for execution_timer"""
    def __init__(self):
        self.log = []
        self.running = False
        self.loop = {}
        
    def _pretty_time(self, time):
        cutoffs = [0.000000001, 0.000001, 0.001, 1, 60,
                   60*60, 60*60*24, 60*60*24*365]

        units = {cutoffs[0]: 'ps',
                 cutoffs[1]: 'ns',
                 cutoffs[2]: 'µs',
                 cutoffs[3]: 'ms',
                 cutoffs[4]: 's',
                 cutoffs[5]: 'min',
                 cutoffs[6]: 'h',
                 cutoffs[7]: 'days',
                 }
        
        for i, cutoff in enumerate(cutoffs):
            if time < cutoff:
                time = round(time / cutoffs[i-1], 2)
                return f"{time} {units[cutoff]}"
        else:
            time = round(time / cutoff, 2)
            return f"{time} years"

    def start(self, parameters={}):
        self.loop = {}
        self.loop['parameters'] = parameters
        self.loop['start_time'] = time.time()
        self.running = True

    def stop(self, display_time=True, return_loop=False):
        if ('start_time' not in self.loop.keys()) or not self.running:
            raise CustomException("You need to start a timer, "
                                  "before you can stop one.")
        self.loop['stop_time'] = time.time()
        self.loop['time'] = (self.loop['stop_time'] - self.loop['start_time'])
        self.loop['pretty_time'] = self._pretty_time(self.loop['time'])

        if display_time:
            print(f"Execution time: {self.loop['pretty_time']}")

        self.log.append(self.loop)
        self.running = False

        if return_loop:
            return self.loop

    def clear(self):
        self.log = []

    def log_df(self):
        if len(self.log) < 1:
            return None
        
        dflog = copy.deepcopy(self.log)

        for i, line in enumerate(dflog):
            if type(line['parameters']) == dict:
                for key in line['parameters'].keys():
                    if key in dflog[i].keys():
                        key += "_parameter"
                    dflog[i][key] = line['parameters'][key]

        df = pd.DataFrame(dflog)
        df = df.drop('parameters', axis=1)
        return df

    def plot(self, parameters, log=False):
        df = self.log_df()

        if type(parameters) != list:
            parameters = [parameters]

        for parameter in parameters:
            g = sns.relplot(data=df, x=parameter, y="time", kind='scatter')
            g.map_dataframe(sns.lineplot, parameter, 'time')
            if log:
                g.set(xscale="log")
                g.set(yscale="log")

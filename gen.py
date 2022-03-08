'''
General Purpose Tools
'''

import re
import os
import shutil


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
    for i, file_path in enumerate(file_list):
        file_name = os.path.basename(file_path)
        new_path = os.path.join(destination_path, file_name)
        if os.path.isfile(new_path):
            if message:
                print(f"{i}. Exists: {os.path.basename(file_name)}")
        else:
            shutil.copy(file_path, new_path)
            if message:
                print(f"{i}. Copied: {os.path.basename(file_name)}")

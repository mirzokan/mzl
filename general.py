'''
General Purpose Tools
'''

import re
import os
import shutil
import time
import pandas as pd
import seaborn as sns
import copy
from typing import Any, Optional, List, Union


class CustomException(Exception):
    """Custom exception for general-purpose tools."""
    pass


def sdir(obj: Any, sunder: bool = False) -> None:
    """
    Enhanced dir function to inspect callable and non-callable attributes.

    Args:
        obj (Any): The object to inspect.
        sunder (bool): If True, includes single-underscore-prefixed attributes.
    """
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


def subl(a: List[Any], b: List[Any]) -> List[Any]:
    """
    Subtract list b from list a.

    Args:
        a (List[Any]): List from which to subtract.
        b (List[Any]): Elements to subtract from list a.

    Returns:
        List[Any]: Resulting list after subtraction.
    """
    return [x for x in a if x not in b]


def get_latest_file(path: str, pattern: Optional[str] = None) -> str:
    """
    Return the most recent timestamped (last alphabetically) file from 
    a folder.

    Args:
        path (str): Directory path to scan.
        pattern (Optional[str]): Optional regex pattern to filter 
                                 filenames.

    Returns:
        str: Full path to the most recent file.

    Raises:
        CustomException: If path does not exist.
        FileNotFoundError: If no matching file is found.
    """
    if not os.path.exists(path):
        raise CustomException('Specified path not found')

    files = [name for name in os.listdir(path) 
             if os.path.isfile(os.path.join(path, name))]
    if pattern is not None:
        files = [file for file in files 
                 if re.fullmatch(pattern, file) is not None]
    files.sort(reverse=True)
    if len(files) >= 1:
        latest_file = os.path.join(path, files[0])
    else:
        print(f"path:{path}")
        print(f"files:{files}")
        raise FileNotFoundError
    return latest_file


def list_files_by_ext(path: str,
                      ext: str,
                      excluded_subfolder_terms: Optional[List[str]] = None) -> List[str]:
    """
    Recursively list all files with a given extension, excluding folders
    by keyword.

    Args:
        path (str): Root directory to search.
        ext (str): File extension filter (e.g., '.csv').
        excluded_subfolder_terms (Optional[List[str]]): Substrings to exclude
                                                        folders.

    Returns:
        List[str]: List of matching file paths.
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


def copy_files(file_list: List[str],
               destination_path: str,
               message: bool = False) -> None:
    """
    Copy files to a destination directory without overwriting existing ones.

    Args:
        file_list (List[str]): Paths to files to copy.
        destination_path (str): Target directory.
        message (Union[bool, str], optional): If True, prints copied or 
                                              skipped files.
    """
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


class execution_timer(object):
    """
    Utility class for measuring code execution time.

    Attributes:
        log (List[dict]): List of recorded timing entries.
        running (bool): Indicates if a timer is currently running.
        loop (dict): Current loop's timing info.
    """
    def __init__(self) -> None:
        self.log: List[dict] = []
        self.running: bool = False
        self.loop: dict = {}
        
    def _pretty_time(self, time_value: float) -> str:
        """
        Convert a raw time value into a human-readable format.

        Args:
            time_value (float): Time in seconds.

        Returns:
            str: Formatted time string.
        """
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

    def start(self, parameters: dict = {}) -> None:
        """
        Start timing.

        Args:
            parameters (dict): Optional parameters to log with the timing.
        """
        self.loop = {}
        self.loop['parameters'] = copy.deepcopy(parameters)
        self.loop['start_time'] = time.time()
        self.running = True

    def stop(self,
             display_time: bool = True,
             return_loop: bool = False) -> Optional[dict]:
        """
        Stop timing and optionally print or return the result.

        Args:
            display_time (bool): Print the elapsed time if True.
            return_loop (bool): Return timing info if True.

        Returns:
            Optional[dict]: Timing log of current loop if return_loop is True.

        Raises:
            CustomException: If stop is called without starting.
        """
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

    def clear(self) -> None:
        """Clear all stored timing logs."""
        self.log = []

    def log_df(self) -> Optional[pd.DataFrame]:
        """
        Return the timing logs as a pandas DataFrame.

        Returns:
            Optional[pd.DataFrame]: DataFrame of timing logs, or None if empty.
        """
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

    def plot(self, parameters: Union[str, List[str]],
             log: bool = False) -> None:
        """
        Plot execution times against one or more parameters.

        Args:
            parameters (Union[str, List[str]]): Parameter(s) to plot on x-axis.
            log (bool): Whether to use log scale.
        """
        df = self.log_df()

        if type(parameters) != list:
            parameters = [parameters]

        for parameter in parameters:
            g = sns.relplot(data=df, x=parameter, y="time", kind='scatter')
            g.map_dataframe(sns.lineplot, parameter, 'time')
            if log:
                g.set(xscale="log")
                g.set(yscale="log")


def parse_replace(input_string: str,
                  start_delim: str = "(",
                  end_delim: str = ")",
                  replacement: str = "",
                  level: int = 1,
                  start_alt: Optional[str] = None,
                  end_alt: Optional[str] = None,
                  strict_eos: bool = False) -> str:
    """
    Replace delimiters at a specific nesting level in a string.

    Args:
        input_string (str): Text to parse.
        start_delim (str): Start delimiter (default: "(").
        end_delim (str): End delimiter (default: ")").
        replacement (str): Replacement for the delimiters (default: "").
        level (int): Nesting level at which to replace delimiters.
        start_alt (Optional[str]): Alternative start delimiter 
                                   (ignored for replacement).
        end_alt (Optional[str]): Alternative end delimiter 
                                 (ignored for replacement).
        strict_eos (bool): If True, alternative end only applies at string end.

    Returns:
        str: Modified string with delimiters replaced.

    Raises:
        Exception: On invalid delimiter configuration.
    """
    
    if start_delim == end_delim:
        raise Exception("Delimeters must be different.")
       
    if ((start_alt is not None and len(start_alt) > len(start_delim)) or
       (end_alt is not None and len(end_alt) > len(end_delim))): 
        raise Exception("Alternative delimiter must not "
                        "be longer than the delimiter.")
        
    if start_delim not in input_string:
        return input_string
        
    common_len = 2 if start_delim[0] == end_delim[-1] else 1
    current_level = 0
    chunk_list = []
    rep_len = len(replacement)
    chunk_start = None
    chunk_end = None

    i = 0
    while i <= len(input_string):
        start_del_hit = input_string[i:i+len(start_delim)] == start_delim
        start_alt_hit = input_string[i:i+len(start_alt)] == start_alt
        
        if start_alt_hit and not start_del_hit:
            current_level += 1
        if start_del_hit:
            current_level += 1
            if current_level == level:
                chunk_start = i
            i = i + len(start_delim)
            continue
                    
        end_del_hit = input_string[i:i+len(end_delim)] == end_delim
        end_alt_hit = input_string[i:i+len(end_alt)] == end_alt
        if strict_eos:
            eos_alt_hit = False
        else:
            eos_alt_hit = (input_string[i:i+len(end_alt)] == end_alt and
                           len(input_string[i:].replace(replacement, ""))
                           <= len(end_alt))
        
        if end_alt_hit and not (end_del_hit or eos_alt_hit):
            current_level -= 1
            if current_level <= 0 and chunk_start is not None:
                current_level = 0
                chunk_start = None
                
        if end_del_hit or eos_alt_hit:
            current_level -= 1
            if current_level < 0 and chunk_start is not None:
                current_level = 0
                chunk_start = None
            if current_level+1 == level:            
                chunk_end = i
                if chunk_start is not None and chunk_end is not None:
                    chunk_list.append((chunk_start, chunk_end))
                    chunk_start = None
                    chunk_end = None
            i = i + len(end_delim)-common_len
        i += 1

    output_string = input_string
    for i, chunk in enumerate(chunk_list):
        adjusted_start = chunk[0] + ((rep_len)*2
                                     - len(start_delim)
                                     - len(end_delim))*i
        adjusted_end = chunk[1] + ((rep_len)*2 
                                   - len(start_delim)
                                   - len(end_delim))*i
        
        output_string = (output_string[:adjusted_start] 
                         + replacement 
                         + output_string[adjusted_start
                         + len(start_delim):adjusted_end] 
                         + replacement
                         + output_string[adjusted_end 
                         + len(end_delim):])

    return output_string

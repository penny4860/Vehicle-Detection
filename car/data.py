# -*- coding: utf-8 -*-

import os
import glob
import random

def list_files(directory, pattern="*.*", n_files_to_sample=None, recursive_option=True, random_order=True):
    """list files in a directory matched in defined pattern.

    # Args
        directory : str
            filename of json file
        pattern : str
            regular expression for file matching
        
        n_files_to_sample : int or None
            number of files to sample randomly and return.
            If this parameter is None, function returns every files.
        
        recursive_option : boolean
            option for searching subdirectories. If this option is True, 
            function searches all subdirectories recursively.

    # Returns
        conf : dict
            dictionary containing contents of json file
    """

    if recursive_option == True:
        dirs = [path for path, _, _ in os.walk(directory)]
    else:
        dirs = [directory]
    
    files = []
    for dir_ in dirs:
        for p in glob.glob(os.path.join(dir_, pattern)):
            files.append(p)
    
    if n_files_to_sample is not None:
        if random_order:
            files = random.sample(files, n_files_to_sample)
        else:
            files = files[:n_files_to_sample]
    return files

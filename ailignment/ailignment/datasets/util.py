# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 13:25:46 2021

@author: kiehne
"""

import os


def get_data_path():
    '''
    Utility function to find the "data" folder in the current working dir.
    
    Returns
    -------
    None.

    '''
    # find the containing folder of our own package
    path = __file__
    splits = os.path.normpath(path).split(os.path.sep)
    index = splits.index("ailignment")
    repo = os.path.sep.join(splits[:index])
    data = os.path.join(repo, "data")
    if not os.path.exists(data):
        raise ValueError(f"Failed to find the `data` folder in path `{path}`")
    return data
    


# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 13:47:18 2021

Adapted from:
    https://github.com/hendrycks/ethics/blob/master/utils.py

@author: kiehne
"""
import pandas as pd
import tarfile
import os

from .util import get_data_path

_TASK_MAP={
    "cm":"commonsense",
    "deontology":"deontology",
    "justice":"justice",
    "util":"utilitarianism",
    "virtue":"virtue"
}

def check_data(filename="ethics.tar"):
    '''
    Makes sure that the dataset is extracted and available
    in the data folder
    '''
    data_path = get_data_path()
    tar_path = os.path.join(data_path, filename)
    ethics_path = os.path.join(data_path, "ethics")
    if not os.path.exists(tar_path):
        raise ValueError("You have to download the 'ETHICS' dataset first. Check README")
    if os.path.exists(ethics_path):
        return
    with tarfile.open(tar_path) as t:
        t.extractall(data_path)


def get_ethics(task, split):
    '''
    Loads the ETHICS dataset task and split.
    
    Parameters
    ----------
    task : str
        Either ["cm", "deontology", "justice", "util", "virtue"]
    split : str
        For all tasks ["train", "test", "test_hard"] is valid.
        For task `cm` there is also "ambig" split.

    Returns
    -------
    `pandas.DataFrame` with the task-specific data.

    '''
    check_data()
    folder = _TASK_MAP.get(task,None)
    if folder is None:
        raise ValueError(f"Unknown task '{task}'")
    folder_path = os.path.join(get_data_path(),"ethics")
    folder_path = os.path.join(folder_path,folder)
    filename = f"{task}_{split}.csv"
    if filename not in os.listdir(folder_path):
        raise ValueError(f"Unknown split '{split}' for task '{task}'")
    path = os.path.join(folder + filename)
    data = pd.read_csv(path)
    return data
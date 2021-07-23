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
from collections import defaultdict

_TASK_MAP={
    "cm":"commonsense",
    "deontology":"deontology",
    "justice":"justice",
    "util":"utilitarianism",
    "virtue":"virtue"
}

def check_data():
    '''
    Makes sure that the dataset is extracted and available
    in "data/ethics/"
    '''
    if not os.path.exists("data/ethics.tar"):
        raise ValueError("You have to download the 'ETHICS' dataset first. Check README")
    if os.path.exists("data/ethics/"):
        return
    with tarfile.open("data/ethics.tar") as t:
        t.extractall("data/")


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
    folder = "data/ethics/" + folder + "/"
    filename = f"{task}_{split}.csv"
    if filename not in os.listdir(folder):
        raise ValueError(f"Unknown split '{split}' for task '{task}'")
    path = folder + filename
    data = pd.read_csv(path)
    return data
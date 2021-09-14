# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 14:34:54 2021

@author: kiehne
"""
import zipfile
import pandas as pd
import os

from .util import get_data_path

def get_social_chem101(filename="social-chem-101.zip"):
    '''
    loads the Social Chemistry 101 dataset as a pandas Dataframe.
    Refer to the GitHub of the authors:https://github.com/mbforbes/social-chemistry-101

    Returns
    -------
    None.

    '''
    path = os.path.join(get_data_path(), filename)
    z = zipfile.ZipFile(path)
    tsv = z.open("social-chem-101/social-chem-101.v1.0.tsv")
    data = pd.read_csv(tsv, sep="\t")
    return data
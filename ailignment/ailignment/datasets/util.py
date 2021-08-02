# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 13:25:46 2021

@author: kiehne
"""

import os
from datasets import load_metric
import numpy as np

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
    

def tokenize_and_split(dataset, tokenizer, sentence_col="text"):
    '''
    Takes a `datasets.Dataset` with train and test splits
    and applies the given tokenizer.
    Returns tokenized train and test split datasets
    '''
    def tokenize_function(examples):
        return tokenizer(examples[sentence_col], padding="max_length", truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset["test"]
    return train_dataset, eval_dataset

def join_sentences(dataframe, columns, sep=" [SEP] "):
    '''
    Given a dataframe of strings and a list of columns,
    joins the different columns in the given order using sep.
    '''
    return dataframe[columns].agg(sep.join, axis=1)

def get_accuracy_metric():
    '''
    Loads the accuracy metric for classification with a huggingface training
    pipeline.

    Returns
    -------
    func
        The `compute_metrics` function to pass to Huggingface `Trainer`
        constructor.

    '''
    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    return compute_metrics
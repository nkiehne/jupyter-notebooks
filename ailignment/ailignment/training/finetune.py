# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 13:54:38 2021

@author: nikla
"""

import gc
import torch

from transformers import (
    AutoModelForSequenceClassification, 
     Trainer, TrainingArguments, AutoModelWithLMHead, AutoTokenizer,
)

def clean_up_mem(func=None):
    '''
    Runs python gc to collect all unused objects and then invokes torch to
    delete all unused GPU objects.
    Can be used as a decorator with "@" or just by calling it inline with no
    arguments

    Parameters
    ----------
    func : TYPE, optional
        Will be set automatically when used with the "@" notation.

    Returns
    -------
    None.

    '''
    if func is not None:
        def f(*args, **kwargs):
            func(*args, **kwargs)
            clean_up_mem()
        f.__doc__ = func.__doc__
        return f

    gc.collect()
    torch.cuda.empty_cache()


@clean_up_mem
def sequence_classification(data_func, model, compute_metrics, training_args):
    '''
    Runs a Sequence Classification task with the given data, model, metrics
    and training arguments

    Parameters
    ----------
    data_func : TYPE
        DESCRIPTION.
    model : TYPE
        DESCRIPTION.
    compute_metrics : TYPE
        DESCRIPTION.
    training_args : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForSequenceClassification.from_pretrained(model)

    train, test = data_func(tokenizer)

    trainer = Trainer(model=model, args=training_args, 
                      train_dataset=train, 
                      eval_dataset=test, 
                      compute_metrics=compute_metrics,)
    trainer.train()


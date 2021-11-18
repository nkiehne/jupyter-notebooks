# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 13:54:38 2021

@author: nikla
"""

import gc
import torch

from transformers import (
    AutoModelForSequenceClassification, AutoConfig,
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
            x = func(*args, **kwargs)
            clean_up_mem()
            return x
        f.__doc__ = func.__doc__
        return f

    gc.collect()
    torch.cuda.empty_cache()


@clean_up_mem
def sequence_classification(data_func, model, training_args, compute_metrics=None, use_pretrained=True):
    '''
    Runs a Sequence Classification task with the given data, model, metrics
    and training arguments

    Parameters
    ----------
    data_func : function
        A function taking a tokenizer and returning the training and eval data.
        Their expected type is `datasets.Dataset`.
        TODO: Could also be the tokenized dataset!
    model : str
        The identifier of the model you would like to train on.
    compute_metrics : function
        The metrics to evaluate on, if desired.
    training_args : `transformers.TrainingArguments`
        The arguments for training, see the corr. doc.
    use_pretrained : bool, True per default
        Whether to use the pre-trained weights or randomly initialize the model.
        In any case, the pre-trained tokenizer will be used!
    Returns
    -------
    A list of the training history. Note, that all logs are in a single list,
    disregarding whether belonging to eval or training.

    '''
    tokenizer = AutoTokenizer.from_pretrained(model)
    if use_pretrained:
        model = AutoModelForSequenceClassification.from_pretrained(model)
    else:
        config = AutoConfig.from_pretrained(model)
        model = AutoModelForSequenceClassification.from_config(config)

    train, test = data_func(tokenizer)

    trainer = Trainer(model=model, args=training_args, 
                      train_dataset=train, 
                      eval_dataset=test, 
                      compute_metrics=compute_metrics,)
    trainer.train()
    return trainer.state.log_history


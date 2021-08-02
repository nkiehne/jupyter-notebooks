# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 16:01:16 2021

@author: nikla
"""

from ailignment.datasets.moral_stories import get_moral_stories, make_action_classification_dataframe
from ailignment.datasets import get_accuracy_metric, join_sentences, tokenize_and_split
from transformers import TrainingArguments
import pandas as pd
import datasets
from ailignment import sequence_classification

def t():
    dataframe = get_moral_stories()
    test_split = 0.2
    batch_size = 8
    model = "distilbert-base-uncased"
    #model = "albert-base-v2"
    action_dataframe = make_action_classification_dataframe(dataframe)
    input_columns = ["action", "consequence"]
    action_dataframe["task_input"] = join_sentences(action_dataframe, input_columns, " ")
    dataset = datasets.Dataset.from_pandas(action_dataframe)
    dataset = dataset.train_test_split(test_size=test_split)

    def data_all(tokenizer):
        return tokenize_and_split(dataset, tokenizer, "task_input")
    def data_small(tokenizer):
        train, test = data_all(tokenizer)
        train = train.shuffle(seed=42).select(range(500))
        test = test.shuffle(seed=42).select(range(500))
        return train, test

    training_args = TrainingArguments(
        output_dir="results/",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        save_steps=1000,
        save_total_limit=0,
        evaluation_strategy="epoch",
    )

    sequence_classification(data_small, model, get_accuracy_metric(), training_args)
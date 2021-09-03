# Data loaders returning `pandas` objects
import pandas as pd
import json
import os

from .util import get_data_path

def get_moral_stories(filename = "moral_stories_datasets.tar.xz"):
    '''
    Loads the `Moral-Stories` dataset by Emelin et al.
    Github & paper: https://github.com/demelin/moral_stories
    Download link for the dataset: https://tinyurl.com/y99sg2uq

    Returns
    -------
    data : pandas.DataFrame
        Columns:
            ID                     object
            norm                   object
            situation              object
            intention              object
            moral_action           object
            moral_consequence      object
            immoral_action         object
            immoral_consequence    object
            dtype: object
        All values are string. No NaNs.

    '''
    path = os.path.join(get_data_path(), filename)
    data = pd.read_csv(path, compression="xz", sep="\t", converters={0:json.loads})
    data = pd.DataFrame(list(data["moral_stories_datasets/"]))
    # filter out the 700k superfluous NaN entries
    data = data[data["label"].isna()]
    data = data.drop(["label"], axis=1)
    return data

def make_action_classification_dataframe(dataframe):
    '''
    Returns a dataframe with columns:
    `norm`, `action`, `situation`, `intention`, `consequence`,
    and `label`.
    The following columns differ from the original definition
    as returned by `get_moral_stories`:
        `action`: Is either the `moral_action` or `immoral_action`
        `label`: 1 if `action` is moral, 0 else
        `consequence`: The corresponding consequence of an action
    '''

    immoral_df = dataframe.drop(["moral_action", "moral_consequence"], axis=1)
    moral_df = dataframe.drop(["immoral_action", "immoral_consequence"], axis=1)
    # rename columns
    moral_df.rename(columns={"moral_action":"action",
                            "moral_consequence":"consequence"},
                    inplace=True)
    immoral_df.rename(columns={"immoral_action":"action",
                            "immoral_consequence":"consequence"},
                    inplace=True)
    # add labels
    immoral_df["labels"] = 0
    moral_df["labels"] = 1
    
    data = pd.concat([moral_df, immoral_df], ignore_index=True)
    return data

import spacy

from string import punctuation
from nltk.corpus import stopwords
from collections import Counter


def _lemmatize(line, nlp, STOP_WORDS=None):
    """ Helper function for obtaining various word representations """
    '''
    Copied from Moral Stories GitHub: https://github.com/demelin/moral_stories/blob/master/split_creation/create_action_lexical_bias_splits.py
    Authors: Emelin et al.
    Minor edits by me.
    '''
    # strip, replace special tokens
    orig_line = line
    line = line.strip()
    # Remove double space
    line = ' '.join(line.split())
    # Tokenize etc.
    line_nlp = nlp(line)
    spacy_tokens = [elem.text for elem in line_nlp]
    spacy_tokens_lower = [elem.text.lower() for elem in line_nlp]
    spacy_lemmas = list()
    for elem in line_nlp:
        if elem.lemma_ == '-PRON-' or elem.lemma_.isdigit():
            spacy_lemmas.append(elem.lower_)
        else:
            spacy_lemmas.append(elem.lemma_.lower().strip())

    # Generate a mapping between whitespace tokens and SpaCy tokens
    ws_tokens = orig_line.strip().split()
    ws_to_spacy_map = dict()
    spacy_to_ws_map = dict()
    ws_loc = 0
    ws_tok = ws_tokens[ws_loc]
    '''
    for spacy_loc, spacy_tok in enumerate(spacy_tokens):
        while True:
            # Map whitespace tokens to be identical to spacy tokens
            if spacy_tok == ws_tok or spacy_tok in ws_tok:
                # Terminate
                if ws_loc >= len(ws_tokens):
                    break

                # Extend maps
                if not ws_to_spacy_map.get(ws_loc, None):
                    ws_to_spacy_map[ws_loc] = list()
                ws_to_spacy_map[ws_loc].append(spacy_loc)
                if not spacy_to_ws_map.get(spacy_loc, None):
                    spacy_to_ws_map[spacy_loc] = list()
                spacy_to_ws_map[spacy_loc].append(ws_loc)

                # Move pointer
                if spacy_tok == ws_tok:
                    ws_loc += 1
                    if ws_loc < len(ws_tokens):
                        ws_tok = ws_tokens[ws_loc]
                else:
                    ws_tok = ws_tok[len(spacy_tok):]
                break
            else:
                ws_loc += 1

    # Assert full coverage of whitespace and SpaCy token sequences by the mapping
    ws_covered = sorted(list(ws_to_spacy_map.keys()))
    spacy_covered = sorted(list(set(list([val for val_list in ws_to_spacy_map.values() for val in val_list]))))
    assert ws_covered == [n for n in range(len(ws_tokens))], \
        'WS-SpaCy mapping does not cover all whitespace tokens: {}; number of tokens: {}'\
        .format(ws_covered, len(ws_tokens))
    assert spacy_covered == [n for n in range(len(spacy_tokens))], \
        'WS-SpaCy mapping does not cover all SpaCy tokens: {}; number of tokens: {}' \
        .format(spacy_covered, len(spacy_tokens))
    '''
    if STOP_WORDS is not None:
        # Filter out stopwords
        nsw_spacy_lemmas = list()
        for tok_id, tok in enumerate(spacy_tokens_lower):
            if tok not in STOP_WORDS:
                nsw_spacy_lemmas.append(spacy_lemmas[tok_id])
            else:
                nsw_spacy_lemmas.append('<STPWRD>')

        spacy_lemmas = nsw_spacy_lemmas

    return spacy_lemmas, ws_tokens, spacy_to_ws_map


def _lemmatize_series(series, nlp, STOP_WORDS=None):
    '''
    Given a series of strings, returns a DataFrame(["lemmas", "tokens", "maps"])
    of the lemmatized strings according to `_lemmatize` function.
    '''
    translation_table = str.maketrans(' ', ' ', punctuation)
    series = series.map(lambda x: x.translate(translation_table))
    series = series.map(lambda x: _lemmatize(x, nlp, STOP_WORDS))
    data = pd.DataFrame(series.to_list(), columns=["lemmas", "tokens", "maps"])
    return data

def _get_lexical_bias_split(columns, top_n, split_size):
    '''
    Given two columns of the moral stories dataset, computes how biased the
    lemmas of the values are towards each of the columns. It draws the top_n
    most biased lemmas and counts for each story, how many occur in the
    corresponding column.
    Returns the moral stories dataset in ascending order of biasness:
        test set (len split_size)
        val set (len split_size)
        train set (rest of the stories)
    '''
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'textcat'])
    STOP_WORDS = stopwords.words('english')
    stories = get_moral_stories()
    
    data = [_lemmatize_series(stories[c], nlp, STOP_WORDS) for c in columns]
    
    # count lemmas per column
    counts = [Counter() for i in data]
    for d,c in zip(data, counts):
        d["lemmas"].apply(c.update)
        c.pop("<STPWRD>")
    
    # compute frequency diffs between words
    if len(counts) != 2: raise ValueError("We are only working with 2 columns here")
    left, right = counts
    all_lemmas = set(left.keys()).union(right.keys())
    freq_diffs = [(l, left.get(l,0) - right.get(l,0)) for l in all_lemmas]
    freq_diffs = sorted(freq_diffs, key=lambda x: abs(x[1]), reverse=True)[:top_n]
    
    # count how many biased lemmas are in the columns
    freq_lemmas = [[x[0] for x in freq_diffs if x[1] >=0], [x[0] for x in freq_diffs if x[1] <0]]
    freq_lemmas = [set(x) for x in freq_lemmas]
    stories["biasness"] = sum([d["lemmas"].map(\
        lambda x: len(f.intersection(x))) for d,f in zip(data, freq_lemmas)])
    # sort by bias-ness in ascending order
    stories = stories.sort_values("biasness")
    
    # create splits
    test = stories[:split_size]
    val = stories[split_size:split_size*2]
    train = stories[split_size*2:]
    return train, val, test

def get_action_lexical_bias_split(top_n = 100, split_size=1000):
    '''
    Generates the Lexical Bias Split for the Actions.

    Parameters
    ----------
    top_n : int, optional
        The number of most frequent lemmas to consider. The default is 100.
    split_size : int, optional
        Size of the dev / test set. The default is 1000.

    Returns
    -------
    None.

    '''
    columns = ["moral_action", "immoral_action"]
    return _get_lexical_bias_split(columns, top_n, split_size)

def get_consequence_lexical_bias_split(top_n = 100, split_size=1000):
    '''
    Generates the Lexical Bias Split for the Consequences.

    Parameters
    ----------
    top_n : int, optional
        The number of most frequent lemmas to consider. The default is 100.
    split_size : int, optional
        Size of the dev / test set. The default is 1000.

    Returns
    -------
    None.

    '''
    columns = ["moral_consequence", "immoral_consequence"]
    return _get_lexical_bias_split(columns, top_n, split_size)

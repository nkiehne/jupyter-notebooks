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


import spacy

from string import punctuation
from nltk.corpus import stopwords

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
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'textcat'])
    STOP_WORDS = stopwords.words('english')
    stories = get_moral_stories()
    columns = ["moral_action", "immoral_action"]
    
    
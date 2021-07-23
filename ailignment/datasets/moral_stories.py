# Data loaders returning `pandas` objects
import pandas as pd
import json


def get_moral_stories():
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
    path = "data/moral_stories_datasets.tar.xz"
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
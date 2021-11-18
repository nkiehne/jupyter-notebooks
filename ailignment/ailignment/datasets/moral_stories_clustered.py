# Data loaders returning `pandas` objects
# The methods in this file are similar to the ones in the moral_stories.py,
# but they allow for additional cluster parametrization, e.g. for better splitting.
# The problem behind all this is, that multiple norms can have similar or equal meaning,
# and if only half of them are flipped, the resulting dataset is very inconsistent.
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from ailignment.training.finetune import clean_up_mem
from ailignment.datasets.moral_stories import randomize_norm_value, flip_norm

@clean_up_mem
def assign_norm_clusters(dataframe, embedding_model='all-distilroberta-v1'):
    '''
    Clusters the "norm_action" in the given dataframe according
    to sentence_embeddings from transformer models.
    A new column "cluster" will be assigned to the dataframe.
    KMeans will be employed.
    '''
    model = SentenceTransformer(embedding_model)
    embeddings = model.encode(dataframe["norm_action"], show_progress_bar=True)
    clustering = KMeans(n_clusters=100, init="k-means++", max_iter=300, n_init=5)
    dataframe["cluster"] = clustering.fit_predict(embeddings)
    return dataframe


def make_action_classification_dataframe(dataframe, **split_kwargs):
    '''
    Returns a dataframe with columns:
    `norm`, `action`, `situation`, `intention`, `consequence`,
    and `label`.
    The following columns differ from the original definition
    as returned by `get_moral_stories`:
        `action`: Is either the `moral_action` or `immoral_action`
        `label`: 1 if `action` is moral, 0 else
        `consequence`: The corresponding consequence of an action
        
    Returns training and testing splits of the data according to sklearn's
    GroupShuffleSplit
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
    
    # for splitting, we do not want to split up actions of similar norms.
    # So, we like to keep all norms within a cluster in either train or test.
    gss = GroupShuffleSplit(1, **split_kwargs)
    xi, yi = list(gss.split(dataframe, groups=dataframe["cluster"]))[0]
    train = pd.concat([moral_df.iloc[xi], immoral_df.iloc[xi]], ignore_index=True).sample(frac=1)
    test = pd.concat([moral_df.iloc[yi], immoral_df.iloc[yi]], ignore_index=True).sample(frac=1)
    return train, test


def get_random_value_dataset(dataframe, p=0.5, good_values=None, bad_values=None, top_n=20):
    '''
    Returns a randomized Moral Stories dataset where
    `p`: float
        Controls the probability of a cluster of norms being flipped into
        the opposite judgment (what was good becomes bad)
    `*_values` lists of values from which to sample good or
        bad norm_values, e.g. ["It is good", "it is nice"]
    `top_n`: int, if `*_values` is None, then the top n
        judgments in the dataset are used
    '''
    if good_values is None or bad_values is None:
        # get frequent norm judgments
        top_negative = dataframe.groupby("norm_sentiment").get_group("NEGATIVE")["norm_value"].value_counts()
        top_positive = dataframe.groupby("norm_sentiment").get_group("POSITIVE")["norm_value"].value_counts()
        good_values = top_positive[:top_n].index
        bad_values = top_negative[:top_n].index

    random_values = dataframe.copy()
    random_values = random_values.apply(randomize_norm_value(good_values, bad_values), axis=1)
    # run over all clusters and flip a whole cluster with prob p
    flip_all = flip_norm(good_values, bad_values, 1.1) # a function that flips everything
    flip_none = flip_norm(good_values, bad_values, -1.1) # a function that flips nothing
    def flip_cluster(p):
        def t(data):
            if np.random.rand()<=p:
                return data.apply(flip_all, axis=1)
            return data.apply(flip_none, axis=1)
        return t
    groups = random_values.groupby("cluster").apply(flip_cluster(p))
    return groups
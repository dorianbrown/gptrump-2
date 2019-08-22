#!/usr/bin/env python

import pandas as pd
import numpy as np
import re
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.multiclass import unique_labels


def combine_multipart_messages(df):
    """
    Make sure this is called after message cleaning. We use 'created_at' column,
    and start/end of message to match up ...'s. We also assume dataframe is sorted
    by creation date, ascending downwards.
    :param df:
    :return:
    """

    startswith_dots_regex = r"^\.{2,}"
    endswith_dots_regex = r"\.{2,}$"

    df["first_multipart"] = df.text.str.contains(endswith_dots_regex, regex=True)
    df["second_multipart"] = df.text.str.contains(startswith_dots_regex, regex=True)
    first_part_w_match = ((df.first_multipart == df.second_multipart.shift(-1)) & df.first_multipart)
    print(f"Combining {sum(first_part_w_match)} multipart messages")

    for ind in np.where(first_part_w_match)[0]:
        if ind not in df.index:
            continue
        text = re.sub(endswith_dots_regex, " ", df.loc[ind, "text"])

        # Needed for messages with >2 parts
        drop_ind = ind
        while df.loc[drop_ind + 1, "second_multipart"]:
            drop_ind += 1
            text += re.sub(endswith_dots_regex, " ",
                           re.sub(startswith_dots_regex, "", df.loc[drop_ind, "text"]))
            df = df.drop(drop_ind)

        df.loc[ind, "text"] = text
    return df


def preprocessing(df):
    start_length = len(df)
    df = df.copy()

    # Removing retweets (not_trump)
    df = df[~df.is_retweet.isna()]
    is_bool = df.is_retweet.apply(lambda x: type(x) == bool)
    df.loc[~is_bool, "is_retweet"] = df.loc[~is_bool, "is_retweet"].apply(lambda x: True if x == "true" else False)
    df.loc[:, "is_retweet"] = df.loc[:, "is_retweet"].astype(bool)
    df = df[~df.is_retweet]
    # Removing rows where text is NA
    df = df[~df.text.isna()]

    # Removing URLs from messages
    url_regex = r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})"
    df.text = df.text.str.replace(url_regex, "", regex=True)
    # Issue with twitter ampersand encoding
    df.text = df.text.str.replace("&amp;", "&")

    # Filtering out rows where the datetime is messed up
    df = df[df.created_at.apply(type) == type('')]
    df = df[df.created_at.apply(len) == 19]

    # Checking format string
    if sum(df.created_at.str.contains("^[0-9]{4}", regex=True)) > 0.9*len(df):
        format_string = "%Y-%m-%d %H:%M:%S"
    else:
        format_string = "%m-%d-%Y %H:%M:%S"
    df.created_at = pd.to_datetime(df.created_at, format=format_string)
    # remove empty text
    print(f"Removed {start_length - len(df)} rows during preprocessing")

    # Making sure df is sorted by creation time, ascending
    df = df.sort_values("created_at")
    df = df.reset_index().drop("index", axis=1)

    df = combine_multipart_messages(df)

    df['is_trump'] = False
    df.loc[df['source'] == 'Twitter for Android', 'is_trump'] = True

    return df


class TextFeatures(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, text):
        no_punc = text.str.replace(r"[^\w\s]", "", regex=True)
        matches = no_punc.str.findall(r"\s([A-Z]+)\s")
        return_df = pd.DataFrame({
            'cap_words': matches.apply(lambda x: len([m for m in x if len(x) > 1]))
        })
        return return_df


class DatetimeFeatures(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, datetimes):
        return_df = pd.DataFrame({
            'dayofweek': datetimes.dt.dayofweek,
            'hour': datetimes.dt.hour,
            'minute': datetimes.dt.minute
        })
        return return_df


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


def create_train_test(df):
    df = df.sort_values("created_at")
    X_train, X_test, y_train, y_test = train_test_split(df.drop('is_trump', axis=1), df['is_trump'],
                                                        test_size=0.25, shuffle=False)
    print(f"Train")
    print(f"not_trump:trump {sum(~y_train)}:{sum(y_train)}")
    print(f"Test")
    print(f"not_trump:trump {sum(~y_test)}:{sum(y_test)}")
    return X_train, X_test, y_train, y_test


def evaluate_model(X_test, y_test, clf, threshold=0.5):
    print(f"Evaluating with threshold: {threshold}")
    y_pred = clf.predict_proba(X_test)[:, 1] > threshold
    # print AUC
    print(f"AUC: {roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]):.3f}")
    # print Accuracy
    print(f"TN: {sum(~y_test & ~y_pred)} FP: {sum(~y_test & y_pred)}")
    print(f"FN: {sum(y_test & ~y_pred)} TP: {sum(y_test & y_pred)}")

    # plot distributions of probabilities
    probs = clf.predict_proba(X_test)[:, 0]
    sns.distplot(probs[y_test == 0], color='g', norm_hist=False)
    sns.distplot(probs[y_test == 1], color='r', norm_hist=False)
    plt.axvline(x=threshold, color="orange")
    plt.show()
    pass
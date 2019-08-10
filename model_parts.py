#!/usr/bin/env python

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np


def preprocessing(df):
    # makesure all datetimes are correct string format
    # convert datetimes
    # remove empty text
    pass

def create_train_test(df):
    df['is_trump'] = False
    df.iloc[df['source'] == 'Twitter for Android', 'is_trump'] = True
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['is_trump'])
    return X_train, X_test, y_train, y_test

def evaluate_model(X_test, y_test, clf):
    # print AUC
    # print Accuracy
    # print confusion matrix
    # plot distributions of probabilities
    pass

def explain_docs():
    pass

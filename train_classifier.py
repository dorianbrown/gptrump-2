#!/usr/bin/env python

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score
from datetime import datetime
import numpy as np

import model_parts

# Set seed here
np.random.seed(1234)


if __name__ == "__main__":

    # Loading
    twitter_df = pd.read_csv("trump_twitter.csv")
    print(twitter_df.head())
    # Preprocessing
    twitter_df = model_parts.preprocessing(twitter_df)

    # Creating labels
    X_train, X_test, y_train, y_test = model_parts.create_train_test(twitter_df)

    print(f"Train: {len(X_train)}")
    print(f"Test: {len(X_test)}")

    clf = Pipeline(steps=[('tfidf', TfidfVectorizer(stop_words='english',
                                                    ngram_range=(1, 2))),
                          ('classifier', LogisticRegression())])

    clf.fit(X_train, y_train)
    model_parts.evaluate_model(X_test, y_test, clf)
    # Try Glove and FastText for features

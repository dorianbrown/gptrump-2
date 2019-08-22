#!/usr/bin/env python

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
import numpy as np
from datetime import datetime
import joblib

import model_parts
import utils
from model_parts import ItemSelector, DatetimeFeatures, TextFeatures


# Set seed here
np.random.seed(1234)


if __name__ == "__main__":

    print("\n# LOADING #\n")
    fname = "data/trump_twitter_200905_201607.csv"
    twitter_df = pd.read_csv(fname)
    print(f"Loaded {len(twitter_df)} rows from \"{fname}\"")

    print("\n# PREPROCESSING #\n")
    twitter_df = model_parts.preprocessing(twitter_df)
    X_train, X_test, y_train, y_test = model_parts.create_train_test(twitter_df)

    clf = Pipeline([
        ('union', FeatureUnion(
            transformer_list=[
                # ('text_feat', Pipeline([
                #     ('selector', ItemSelector('text')),
                #     ('text_feat', TextFeatures())
                # ])),
                ('bow_text', Pipeline([
                    ('selector', ItemSelector('text')),
                    ('tfidf', TfidfVectorizer(stop_words='english',
                                              ngram_range=(1, 1)))
                ])),
                ('dt_feat', Pipeline([
                    ('selector', ItemSelector('created_at')),
                    ('dt_feat', DatetimeFeatures())
                ]))
            ]
        )),
        ('classifier', LogisticRegression(solver='lbfgs',
                                          max_iter=1000))
    ])

    print("\n# TRAINING #\n")
    clf.fit(X_train, y_train)
    model_parts.evaluate_model(X_test, y_test, clf, threshold=0.8)
    output_fname = "models/trump_classifier.pkl"
    print(f"Saving model to {output_fname}")
    joblib.dump(clf, output_fname)


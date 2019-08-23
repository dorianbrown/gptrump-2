#!/usr/bin/env python

import pandas as pd
import joblib

import model_parts
import utils

if __name__ == "__main__":
    output_twitter_df = pd.read_csv("data/trump_twitter_201601_201908.csv")
    print(f"Loaded {len(output_twitter_df)} rows")
    clf = joblib.load("models/trump_classifier.pkl")

    output_twitter_df = model_parts.preprocessing(output_twitter_df)
    is_trump = clf.predict(output_twitter_df)
    print(f"Predicted labels: not_trump:trump {sum(~is_trump)}:{sum(is_trump)}")

    print("\n# NOT_TRUMP Predictions\n")
    n = 5
    utils.print_messages(output_twitter_df[~is_trump].sample(n)["text"])
    print("\n# TRUMP Predictions\n")
    utils.print_messages(output_twitter_df[is_trump].sample(n)["text"])

    output_twitter_df[["text"]].to_csv("output/real_trump_twitter_201601_201908.csv", index=False)
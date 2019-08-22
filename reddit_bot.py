#! /usr/bin/env python3
import os
import time
import datetime
import secrets
from praw.exceptions import *
import time

import praw
import fire

from finish_message import finish_message


# TODO: Add logger for print statements
# TODO: Work on username mentions. using praw.inbox() functionality.

def flatten(nested_lst):
    if not isinstance(nested_lst, list):
        return nested_lst

    res = []
    for l in nested_lst:
        if not isinstance(l, list):
            res += [l]
        else:
            res += flatten(l)

    return (res)


def get_comments(submission):
    def get_replies(comment):
        return [comment] + [get_replies(reply) for reply in comment.replies.list()]

    submission.comments.replace_more(limit=None)
    return set(flatten([get_replies(comment) for comment in submission.comments]))

def run_bot(
        debug=True,
        subreddit="testingground4bots"
):
    USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36"
    DEBUG = debug
    if DEBUG:
        SUB = "testingground4bots"
    else:
        SUB = subreddit

    MESSAGE_TEMPLATE = """{} \n
    I am a bot. Message me to contact my creator. \n
    I am open source. [Fork me on Github](https://github.com/BlitzKraft/AreYouSureItsNotSMBC)
    """

    reddit = praw.Reddit(client_id=secrets.CLIENT_ID,
                         client_secret=secrets.CLIENT_SECRET,
                         user_agent=USER_AGENT,
                         username=secrets.REDDIT_USER,
                         password=secrets.REDDIT_PASSWORD)

    if not os.path.isfile("replied_to.txt"):
        posts_replied = []
    else:
        with open("replied_to.txt", "r") as f:
            posts_replied = f.read()
            posts_replied = posts_replied.split("\n")
            posts_replied = list(filter(None, posts_replied))

    while True:
        for submission in reddit.subreddit(SUB).new(limit=20):
            print(f"Submission: {submission.title}")
            for comment in get_comments(submission):
                if "GOGO TRUMPBOT" in comment.body:
                    parent = comment.parent()
                    text_to_complete = parent.body.strip().split("\n")
                    print("Generating reply")

                    i = 0
                    while True:
                        i += 1
                        model_completion = finish_message(raw_text=text_to_complete)
                        completion_sentences = model_completion.split("\n")
                        return_sentence = completion_sentences[0].strip()
                        if (len(return_sentence) > 200) or (i > 20):
                            break

                    try:
                        print(f"Submitting reply to {submission}")
                        parent.reply(MESSAGE_TEMPLATE.format(return_sentence))
                    except APIException as e:
                        with open("auto_smbc_bot.log", "a") as f:
                            f.write('{:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now()) + ": API Exception. "
                                                                                             "Probably a rate "
                                                                                             "limit\n");

        time.sleep(10)


if __name__ == "__main__":
    fire.Fire(run_bot)

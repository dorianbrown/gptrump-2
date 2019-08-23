#! /usr/bin/env python3
import os
import textwrap
import datetime
import secrets
from praw.exceptions import *
import time

import praw
import fire

from gptrump2 import GPTrump2


# TODO: Add logger for print statements


def run_bot(debug=True, subreddit="testingground4bots"):

    USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36"
    DEBUG = debug

    MESSAGE_TEMPLATE = textwrap.dedent("""\
    > **{}** {}
    
    *****
    I'm a bot | Summon me by mentioning u/trump_rantbot | Generating using [gpt-2](https://github.com/openai/gpt-2)""")

    reddit = praw.Reddit(client_id=secrets.CLIENT_ID,
                         client_secret=secrets.CLIENT_SECRET,
                         user_agent=USER_AGENT,
                         username=secrets.REDDIT_USER,
                         password=secrets.REDDIT_PASSWORD)

    gptrump2 = GPTrump2(model_name="117M1000Iter")

    if not os.path.isfile("replied_to.txt"):
        posts_replied = []
    else:
        with open("replied_to.txt", "r") as f:
            posts_replied = f.read()
            posts_replied = posts_replied.split("\n")
            posts_replied = list(filter(None, posts_replied))

    while True:
        for comment in reddit.inbox.mentions():
            if comment.id not in posts_replied:
                if comment.subreddit.display_name == "testingground4bots" or not DEBUG:
                    parent = comment.parent()
                    text_to_complete = parent.body.strip().split("\n")[-1]
                    print(f"Generating reply to: {text_to_complete}")

                    i = 0
                    while True:
                        i += 1
                        model_completion = gptrump2.complete_text(raw_text=text_to_complete)
                        completion_sentences = model_completion.split("\n")
                        return_sentence = completion_sentences[0].strip()
                        if (len(return_sentence) > 200) or (i > 20):
                            break

                    try:
                        print(f"Submitting reply to {comment}")
                        print(f"Comment text: {parent.body}")
                        completion = return_sentence[len(text_to_complete):]
                        parent.reply(MESSAGE_TEMPLATE.format(text_to_complete, completion))
                    except APIException as e:
                        with open("auto_smbc_bot.log", "a") as f:
                            f.write('{:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now()) + ": API Exception. "
                                                                                             "Probably a rate "
                                                                                             "limit\n");
                    posts_replied.append(comment.id)
                    with open("replied_to.txt", "a") as f:
                        f.write(comment.id + "\n")

        time.sleep(10)


if __name__ == "__main__":
    fire.Fire(run_bot)

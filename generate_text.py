#/usr/bin/env python

import gpt_2_simple as gpt2

if __name__ == "__main__":

    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess)

    output = gpt2.generate(sess, return_as_list=True)

    for text in output[:10]:
        print(text)

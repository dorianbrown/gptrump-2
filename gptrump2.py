#!/usr/bin/env python3

import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder


class GPTrump2:
    def __init__(
        self,
        model_name='117M1000Iter',
        nsamples=1,
        batch_size=1,
        length=200,
        temperature=0.7,
        top_k=0,
        models_dir='models'
    ):
        models_dir = os.path.expanduser(os.path.expandvars(models_dir))
        if batch_size is None:
            batch_size = 1
        assert nsamples % batch_size == 0

        enc = encoder.get_encoder(model_name, models_dir)
        hparams = model.default_hparams()
        with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
            hparams.override_from_dict(json.load(f))

        if length is None:
            length = hparams.n_ctx // 2
        elif length > hparams.n_ctx:
            raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

        sess = tf.Session()
        context = tf.placeholder(tf.int32, [batch_size, None])
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        self.enc = enc
        self.sess = sess
        self.context = context
        self.batch_size = batch_size
        self.nsamples = nsamples
        self.output = output

    def complete_text(self, raw_text):
        enc = self.enc
        sess = self.sess
        context = self.context
        batch_size = self.batch_size
        nsamples = self.nsamples
        output = self.output

        context_tokens = enc.encode(raw_text)
        generated = 0
        for _ in range(nsamples // batch_size):
            out = sess.run(output, feed_dict={
                context: [context_tokens for _ in range(batch_size)]
            })[:, len(context_tokens):]
            for i in range(batch_size):
                generated += 1
                text = enc.decode(out[i])

        return raw_text + text


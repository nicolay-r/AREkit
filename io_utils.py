#!/usr/bin/python
import io

sent_to_int = {'pos': 1, 'neg': -1, 'neu': 0}
int_to_sent = {1: 'pos', -1: 'neg', 0: 'neu'}


def read_prepositions(filepath):
    prepositions = []
    with io.open(filepath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            prepositions.append(line.strip())

    return prepositions


def train_indices():
    indices = range(1, 46)[:10]
    indices.remove(9)
    return indices


def test_indices():
    indices = range(46, 76)[:10]
    # indices.remove(70)
    return indices


def int_to_sentiment(label):
    return unicode(int_to_sent[label])


def sentiment_to_int(label):
    return sent_to_int[label]


def test_root():
    return "data/test/"


def train_root():
    return "data/Texts/"

#!/usr/bin/python
import io

sent_to_int = {'pos': 1, 'neg': -1, 'neu': 0}
int_to_sent = {1: 'pos', -1: 'neg', 0: 'neu'}
ignored_entity_values = [u"author", u"unknown"]


def read_prepositions(filepath):
    prepositions = []
    with io.open(filepath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            prepositions.append(line.strip())

    return prepositions


def train_indices():
    indices = range(1, 46)
    for i in [9, 22, 26]:
        if i in indices:
            indices.remove(i)
    return indices


def test_indices():
    indices = range(46, 76)
    for i in [70]:
        if i in indices:
            indices.remove(i)
    return indices


def get_ignored_entity_values():
    return ignored_entity_values


def int_to_sentiment(label):
    return unicode(int_to_sent[label])


def sentiment_to_int(label):
    return sent_to_int[label]


def test_root():
    return "data/Test/"


def train_root():
    return "data/Texts/"


def get_etalon_root():
    return "data/Etalon/"


def get_synonyms_filepath():
    return "data/synonyms.txt"

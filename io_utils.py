#!/usr/bin/python
# -*- coding: utf-8 -*-
import io


def read_prepositions(filepath):
    prepositions = []
    with io.open(filepath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            prepositions.append(line.strip())

    return prepositions


def train_indices():
    indices = range(1, 46)[:1]
    # indices.remove(9)
    return indices


def test_indices():
    indices = range(46, 76)[:1]
    # indices.remove(70)
    return indices


def test_root():
    return "data/test/"


def train_root():
    return "data/Texts/"

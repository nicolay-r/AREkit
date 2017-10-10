#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging
import numpy as np
from pymystem3 import Mystem

from core.annot import EntityCollection
from core.news import News
from core.opinion import OpinionCollection

IGNORED_ENTITIES = ["Author", "Unknown"]


# distance between two entities counted in sentences.
def sentences_between(e1, e2, news):

    if e1.value in IGNORED_ENTITIES or e2.value in IGNORED_ENTITIES:
        return -2

    if e1.ID == e2.ID:
        return -1

    s1_ind = None
    s2_ind = None
    for i, s in enumerate(news.sentences):
        if s.has_entity(e1.ID):
            s1_ind = i
            break

    for i, s in enumerate(news.sentences):
        if s.has_entity(e2.ID):
            s2_ind = i
            break

    return abs(s1_ind - s2_ind)


def relations_equal_diff(E, diff, news, opinions):

    def make_relation(e1, e2, mystem):
        return "{}_{}".format(mystem.lemmatize(e1), mystem.lemmatize(e2))

    count = 0
    mystem = Mystem()
    unique_relations = []
    entities = news.entities

    for i in range(E.shape[0]):
        for j in range(E.shape[1]):

            e1 = entities.get(i)
            e2 = entities.get(j)
            r = make_relation(e1.value, e2.value, mystem)
            d = min(abs(e1.end - e2.begin), abs(e2.end - e1.begin))

            if r in unique_relations:
                continue

            if E[i][j] == diff:
                s = opinions.has_opinion(e1.value, e2.value, lemmatize=True)

                # Filter sentiment relations
                if s:
                    continue

                logging.info("{} -> {}, d: {}, s: {}".format(
                    e1.value.encode('utf-8'), e2.value.encode('utf-8'), d, s))

                unique_relations.append(r)
                count += 1

    return count


def id_to_int(e_id):
    assert(type(e_id) == unicode)
    return int(e_id[1:len(e_id)])


annot_filepath = "data/Texts/art2.ann"
news_filepath = 'data/Texts/art2.txt'
opin_filepath = 'data/Texts/art2.opin.txt'

logging.basicConfig(filemode='w',
                    format="",
                    level=logging.DEBUG,
                    filename="out.txt")

entities = EntityCollection.from_file(annot_filepath)
news = News.from_file(news_filepath, entities)
opinions = OpinionCollection.from_file(opin_filepath)

E = np.zeros((entities.count(), entities.count()), dtype='int32')

for e1 in entities:
    for e2 in entities:
        i = id_to_int(e1.ID)
        j = id_to_int(e2.ID)
        E[i-1][j-1] = sentences_between(e1, e2, news)

for diff in range(1):
    logging.info("d: {}, r: {}".format(
        diff, relations_equal_diff(E, diff, news, opinions)))

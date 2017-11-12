#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np

import core.env as env
from core.source.entity import EntityCollection
from core.source.news import News
from core.source.opinion import OpinionCollection, Opinion
from core.source.synonyms import SynonymsCollection

import io_utils

IGNORED_ENTITIES = io_utils.get_ignored_entity_values()


def sentences_between(e1, e2, news):
    """ Distance between two features in sentences
    """
    if e1.ID == e2.ID:
        return -1

    return abs(news.get_sentence_by_entity(e1).index -
               news.get_sentence_by_entity(e2).index)


def opinions_between_entities(E, diff, news, synonyms, sentiment_opins=None):
    """ Relations that had the same difference
    """
    def try_add_opinion(o, added, neutral_opins):
        assert(isinstance(o, Opinion))
        assert(isinstance(neutral_opins, OpinionCollection))

        # Filter if there is a sentiment relation
        if sentiment_opins is not None:
            if sentiment_opins.has_opinion_by_synonyms(o):
                return

        if neutral_opins.has_opinion_by_synonyms(o):
            return

        added.add(o.create_value_id())
        neutral_opins.add_opinion(o)

    def is_ignored(entity):
        # TODO. Move ignored entities into core.
        return env.stemmer.lemmatize_to_str(entity.value) in IGNORED_ENTITIES

    def get_entity_synonyms(entity):
        return synonyms.get_synonyms_list(entity.value), \
               synonyms.get_synonym_group_index(entity.value)

    added = set()
    c = OpinionCollection(opinions=None, synonyms=synonyms)

    for i in range(E.shape[0]):
        for j in range(E.shape[1]):

            if E[i][j] != diff:
                continue

            e1 = news.entities.get_entity_by_index(i)
            e2 = news.entities.get_entity_by_index(j)

            if is_ignored(e1) or is_ignored(e2):
                continue

            if not synonyms.has_synonym(e1.value):
                synonyms.add_synonym(e1.value)

            if not synonyms.has_synonym(e2.value):
                synonyms.add_synonym(e2.value)

            sl1, g1 = get_entity_synonyms(e1)
            sl2, g2 = get_entity_synonyms(e2)

            r_left = sl1[0]
            r_right = sl2[0]

            # Filter the same groups
            if g1 == g2:
                "Entities '{}', and '{}' a part of the same synonym group".format(
                    r_left.encode('utf-8'), r_right.encode('utf-8'))
                continue

            try_add_opinion(Opinion(r_left, r_right, u"neu"), added, c)
            try_add_opinion(Opinion(r_right, r_left, u"neu"), added, c)

    return c


def make_neutrals(news, synonyms_collection, opinions=None):
    entities = news.entities
    E = np.zeros((entities.count(), entities.count()), dtype='int32')
    for e1 in entities:
        for e2 in entities:
            i = e1.get_int_ID()
            j = e2.get_int_ID()
            E[i-1][j-1] = sentences_between(e1, e2, news)

    return opinions_between_entities(
        E, 0, news, synonyms_collection, sentiment_opins=opinions)


#
# Main
#
synonyms = SynonymsCollection.from_file(io_utils.get_synonyms_filepath())

#
# Train
#
root = io_utils.train_root()
for n in io_utils.train_indices():
    entity_filepath = root + "art{}.ann".format(n)
    news_filepath = root + "art{}.txt".format(n)
    opin_filepath = root + "art{}.opin.txt".format(n)
    neutral_filepath = root + "art{}.neut.txt".format(n)

    print neutral_filepath

    entities = EntityCollection.from_file(entity_filepath)
    news = News.from_file(news_filepath, entities)
    opinions = OpinionCollection.from_file(opin_filepath, io_utils.get_synonyms_filepath())

    neutral_opins = make_neutrals(news, synonyms, opinions)
    neutral_opins.save(neutral_filepath)

#
# Test
#
root = io_utils.test_root()
for n in io_utils.test_indices():
    entity_filepath = root + "art{}.ann".format(n)
    news_filepath = root + "art{}.txt".format(n)
    neutral_filepath = root + "art{}.neut.txt".format(n)

    print neutral_filepath

    entities = EntityCollection.from_file(entity_filepath)
    news = News.from_file(news_filepath, entities)

    neutral_opins = make_neutrals(news, synonyms)
    neutral_opins.save(neutral_filepath)


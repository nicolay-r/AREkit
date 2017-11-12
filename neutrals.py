#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np

import core.env as env
from core.source.entity import EntityCollection
from core.source.news import News
from core.source.opinion import OpinionCollection
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


def relations_equal_diff(E, diff, news, synonyms_collection, sentiment_opins=None):
    """ Relations that had the same difference
    """
    def try_add_relation(left_value, right_value, keys, pairs):
        r_key = "%s_%s" % (left_value, right_value)

        if r_key in keys:
            return

        # Filter if there is a sentiment relation
        if sentiment_opins is not None:
            if sentiment_opins.has_opinion_by_values(e1.value, e2.value):
                return

        keys.add(r_key)
        pairs.append((
            env.stemmer.lemmatize_to_str(left_value),
            env.stemmer.lemmatize_to_str(right_value)
        ))

    def is_ignored(entity):
        return env.stemmer.lemmatize_to_str(entity.value) in IGNORED_ENTITIES

    def get_synonyms(entity):
        if not synonyms_collection.has_synonym(entity.value):
            # print "Can't find synonym for '{}'".format(entity.value.encode('utf-8'))
            return [entity.value], None
        return synonyms_collection.get_synonyms(entity.value), \
               synonyms_collection.get_synonym_group_index(entity.value)

    r_keys = set()
    r_pairs = []

    for i in range(E.shape[0]):
        for j in range(E.shape[1]):

            if E[i][j] != diff:
                continue

            e1 = news.entities.get(i)
            e2 = news.entities.get(j)

            if is_ignored(e1) or is_ignored(e2):
                continue

            s_list1, s_group1 = get_synonyms(e1)
            s_list2, s_group2 = get_synonyms(e2)

            r_left = s_list1[0]
            r_right = s_list2[0]

            # Filter the same groups
            if s_group1 is not None and s_group2 is not None:
                if s_group1 == s_group2:
                    "Entities '{}', and '{}' a part of the same synonym group".format(
                        r_left.encode('utf-8'), r_right.encode('utf-8'))
                    continue

            try_add_relation(r_left, r_right, r_keys, r_pairs)
            try_add_relation(r_right, r_left, r_keys, r_pairs)

    print "Relations: '{}'".format(len(r_pairs))
    return r_pairs


def make_neutrals(news, synonyms_collection, opinions=None):
    entities = news.entities
    E = np.zeros((entities.count(), entities.count()), dtype='int32')
    for e1 in entities:
        for e2 in entities:
            i = e1.get_int_ID()
            j = e2.get_int_ID()
            E[i-1][j-1] = sentences_between(e1, e2, news)

    relations = relations_equal_diff(
        E, 0, news, synonyms_collection, sentiment_opins=opinions)

    return relations


def save(filepath, relation_pairs):
    """ Saving relation
    """
    print "save: {}".format(filepath)
    with open(filepath, "w") as f:
        for r_left, r_right in relation_pairs:
            f.write("{}, {}, {}, {}\n".format(
                r_left.encode('utf-8'),
                r_right.encode('utf-8'),
                'neu', 'current'))

#
# Main
#
synonyms_collection = SynonymsCollection.from_file(io_utils.get_synonyms_filepath())

#
# Train
#
root = io_utils.train_root()
for n in io_utils.train_indices():
    print "read: {}".format(n)
    entity_filepath = root + "art{}.ann".format(n)
    news_filepath = root + "art{}.txt".format(n)
    opin_filepath = root + "art{}.opin.txt".format(n)
    neutral_filepath = root + "art{}.neut.txt".format(n)

    entities = EntityCollection.from_file(entity_filepath)
    news = News.from_file(news_filepath, entities)
    opinions = OpinionCollection.from_file(opin_filepath)

    pairs = make_neutrals(news, synonyms_collection, opinions)
    save(neutral_filepath, pairs)

#
# Test
#
root = io_utils.test_root()
for n in io_utils.test_indices():
    print "read: {}".format(n)
    entity_filepath = root + "art{}.ann".format(n)
    news_filepath = root + "art{}.txt".format(n)
    neutral_filepath = root + "art{}.neut.txt".format(n)

    entities = EntityCollection.from_file(entity_filepath)
    news = News.from_file(news_filepath, entities)

    pairs = make_neutrals(news, synonyms_collection)
    save(neutral_filepath, pairs)

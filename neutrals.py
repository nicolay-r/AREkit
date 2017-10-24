#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from gensim.models.word2vec import Word2Vec

import core.environment as env
from core.source.entity import EntityCollection
from core.source.news import News
from core.source.opinion import OpinionCollection

import io_utils

IGNORED_ENTITIES = ["Author", "Unknown"]


def sentences_between(e1, e2, news):
    """ Distance between two features in sentences
    """

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


def relations_equal_diff(E, diff, news, w2v_model, opinions=None):
    """ Relations that had the same difference
    """

    def get_word2vec_vector(lemmas, w2v_model):
        v = np.zeros(w2v_model.vector_size, dtype=np.float32)
        for l in lemmas:
            if l in w2v_model:
                v = v + w2v_model[l]
        return v

    def rus_vectores_similarity(e1, e2, w2v_model):
        l1 = env.stemmer.lemmatize_to_rusvectores_str(e1)
        l2 = env.stemmer.lemmatize_to_rusvectores_str(e2)
        v1 = get_word2vec_vector(l1, w2v_model)
        v2 = get_word2vec_vector(l2, w2v_model)
        return sum(map(lambda x, y: x * y, v1, v2))

    unique_relations = []
    entities = news.entities
    pairs = []

    for i in range(E.shape[0]):
        for j in range(E.shape[1]):

            e1 = entities.get(i)
            e2 = entities.get(j)
            r_left = env.stemmer.lemmatize_to_str(e1.value)
            r_right = env.stemmer.lemmatize_to_str(e2.value)
            r = "%s_%s" % (r_left.encode('utf-8'), r_right.encode('utf-8'))

            if (e1.value in IGNORED_ENTITIES or e2.value in IGNORED_ENTITIES):
                continue

            if r in unique_relations:
                continue

            if E[i][j] != diff:
                continue

            s = False
            if opinions is not None:
                s = opinions.has_opinion(e1.value, e2.value, lemmatize=True)

            # Filter sentiment relations
            if s:
                continue

            unique_relations.append(r)

            pairs.append((r_left, r_right))

    return pairs


def make_neutrals(news, entities, w2v_model, opinions=None):

    E = np.zeros((entities.count(), entities.count()), dtype='int32')
    for e1 in entities:
        for e2 in entities:
            i = e1.get_int_ID()
            j = e2.get_int_ID()
            E[i-1][j-1] = sentences_between(e1, e2, news)

    relations = relations_equal_diff(
        E, 0, news, w2v_model, opinions=opinions)

    return relations


def save(filepath, relation_pairs):
    """ Saving relation
    """
    print filepath
    with open(filepath, "w") as f:
        for r_left, r_right in relation_pairs:
            f.write("{}, {}, {}, {}\n".format(
                r_left.encode('utf-8'),
                r_right.encode('utf-8'),
                'neu', 'current'))

#
# Main
#
w2v_model_filepath = "../tone-classifier/data/w2v/news_rusvectores2.bin.gz"
w2v_model = Word2Vec.load_word2vec_format(w2v_model_filepath, binary=True)

#
# Train
#
root = io_utils.train_root()
for n in io_utils.train_indices():
    entity_filepath = root + "art{}.ann".format(n)
    news_filepath = root + "art{}.txt".format(n)
    opin_filepath = root + "art{}.opin.txt".format(n)
    neutral_filepath = root + "art{}.neut.txt".format(n)

    print entity_filepath

    entities = EntityCollection.from_file(entity_filepath)
    news = News.from_file(news_filepath, entities)
    opinions = OpinionCollection.from_file(opin_filepath)

    pairs = make_neutrals(news, entities, w2v_model, opinions)
    save(neutral_filepath, pairs)

#
# Test
#
root = io_utils.test_root()
for n in io_utils.test_indices():
    entity_filepath = root + "art{}.ann".format(n)
    news_filepath = root + "art{}.txt".format(n)
    neutral_filepath = root + "art{}.neut.txt".format(n)

    entities = EntityCollection.from_file(entity_filepath)
    news = News.from_file(news_filepath, entities)

    pairs = make_neutrals(news, entities, w2v_model)
    save(neutral_filepath, pairs)

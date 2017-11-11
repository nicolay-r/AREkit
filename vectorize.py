#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging
import numpy as np
from gensim.models.word2vec import Word2Vec

import core.environment as env

from core.source.opinion import OpinionCollection
from core.source.entity import EntityCollection
from core.source.news import News
from core.relations import Relation
from core.source.vectors import CommonRelationVectorCollection, CommonRelationVector
from core.source.synonyms import SynonymsCollection

from core.features.distance import DistanceFeature
from core.features.similarity import SimilarityFeature
from core.features.lexicon import LexiconFeature
from core.features.pattern import PatternFeature
from core.features.entities import EntitiesBetweenFeature
from core.features.prepositions import PrepositionsCountFeature
from core.features.frequency import EntitiesFrequency
from core.features.appearance import EntityAppearanceFeature
from core.features.context import ContextPosBeforeFeature, ContextSentimentAfterFeature

from core.processing.prefix import SentimentPrefixProcessor

import io_utils


def is_ignored(entity_value):
    ignored = io_utils.get_ignored_entity_values()
    entity_value = env.stemmer.lemmatize_to_str(entity_value)
    if entity_value in ignored:
        # print "ignored: '{}'".format(entity_value.encode('utf-8'))
        return True
    return False


def vectorize_train(news, entities, opinion_collections, synonym_collection):
    """ Vectorize news of train collection that has opinion labeling
    """
    def get_appropriate_entities(opinion_value):
        if synonyms_collection.has_synonym(opinion_value):
            return filter(
                lambda s: entities.has_enity_by_value(s),
                synonyms_collection.get_synonyms(opinion_value))
        elif entities.has_enity_by_value(opinion_value):
            return [opinion_value]
        else:
            return []

    collection = CommonRelationVectorCollection()
    for opinions in opinion_collections:
        for opinion in opinions:

            # TODO. Later ignore relations between the same entities
            print "{}->{}".format(opinion.entity_left.encode('utf-8'),
                                  opinion.entity_right.encode('utf-8'))
            if opinion.entity_left == opinion.entity_right:
                continue

            left_values = get_appropriate_entities(opinion.entity_left)
            right_values = get_appropriate_entities(opinion.entity_right)

            # TODO. We guarantee that these left and right values are not lemmatized
            if len(left_values) == 0:
                print "Appropriate entity for '{}'->'{}' has not been found".format(
                    opinion.entity_left.encode('utf-8'),
                    env.stemmer.lemmatize_to_str(opinion.entity_left).encode('utf-8')
                )
                continue

            if len(right_values) == 0:
                print "Appropriate entity for '{}' has not been found".format(
                   opinion.entity_right.encode('utf-8'))
                continue

            r_count = 0
            relations = []

            for entity_left in left_values:
                for entity_right in right_values:

                    if is_ignored(entity_left):
                        continue

                    if is_ignored(entity_right):
                        continue

                    entities_left_ids = entities.get_by_value(entity_left)
                    entities_right_ids = entities.get_by_value(entity_right)

                    r_count = len(entities_left_ids) * len(entities_right_ids)

                    # print "{}->{} {}".format(
                    #       opinion.entity_left.encode('utf-8'),
                    #       opinion.entity_right.encode('utf-8'),
                    #       r_count)

                    for e1_ID in entities_left_ids:
                        for e2_ID in entities_right_ids:
                            e1 = entities.get_by_ID(e1_ID)
                            e2 = entities.get_by_ID(e2_ID)
                            r = Relation(e1.ID, e2.ID, news)
                            relations.append(r)

            if r_count == 0:
                continue

            r_features = np.concatenate(
                [f.calculate(relations) for f in FEATURES], axis=0)

            vector = CommonRelationVector(
                opinion.entity_left, opinion.entity_right,
                r_features, io_utils.sentiment_to_int(opinion.sentiment))

            collection.add_vector(vector)

    return collection


def vectorize_test(news, entities):
    """ Vectorize news of test collection
    """
    def create_key(e1, e2):
        key = "{}_{}".format(
            env.stemmer.lemmatize_to_str(e1.value).encode('utf-8'),
            env.stemmer.lemmatize_to_str(e2.value).encode('utf-8')).decode('utf-8')
        assert(type(key) == unicode)
        return key

    collection = CommonRelationVectorCollection()
    relations = {}

    for e1 in entities:
        for e2 in entities:

            if e1.ID == e2.ID:
                continue

            if is_ignored(e1.value):
                continue

            if is_ignored(e2.value):
                continue

            s1 = news.get_sentence_by_entity(e1)
            s2 = news.get_sentence_by_entity(e2)

            if not s1 == s2:
                continue

            r_key = create_key(e1, e2)

            r = Relation(e1.ID, e2.ID, news)
            if r_key not in relations:
                relations[r_key] = []

            relations[r_key].append(r)

    for key, value in relations.iteritems():
        assert(len(value) > 0)
        e1 = entities.get_by_ID(value[0].entity_left_ID)
        e2 = entities.get_by_ID(value[0].entity_right_ID)
        e1_value = env.stemmer.lemmatize_to_str(e1.value)
        e2_value = env.stemmer.lemmatize_to_str(e2.value)

        # print "{}->{}, {}".format(
        #     e1_value.encode('utf-8'),
        #     e2_value.encode('utf-8'),
        #     len(value))

        features = np.concatenate([f.calculate(value) for f in FEATURES], axis=0)
        vector = CommonRelationVector(e1_value, e2_value, features)
        collection.add_vector(vector)

    return collection


def filter_neutral(neutral_opins, news, limit=10):
    scored_opinions = []
    for o in neutral_opins:

        if not entities.has_enity_by_value(o.entity_left):
            scored_opinions.append((o, 0))
            continue

        if not entities.has_enity_by_value(o.entity_right):
            scored_opinions.append((o, 0))
            continue

        entities_left_IDs = len(entities.get_by_value(o.entity_left))
        entities_right_IDs = len(entities.get_by_value(o.entity_right))
        popularity = entities_left_IDs * entities_right_IDs

        scored_opinions.append((o, popularity))

    scored_opinions.sort(key= lambda x: x[1], reverse=True)

    for o, score in scored_opinions[limit:]:
        neutral_opins.remove_opinion(o)


#
# Main
#

root = "data/Texts/"
preps_filepath = "data/prepositions.txt"
rusentilex_filepath = "data/rusentilex.csv"
w2v_model_filepath = "../tone-classifier/data/w2v/news_rusvectores2.bin.gz"
synonyms_filepath = "data/synonyms.txt"

# w2v_model = Word2Vec.load_word2vec_format(w2v_model_filepath, binary=True)
prefix_processor = SentimentPrefixProcessor.from_file("data/prefixes.csv")
prepositions_list = io_utils.read_prepositions(preps_filepath)
synonyms_collection = SynonymsCollection.from_file(synonyms_filepath)

FEATURES = [
    DistanceFeature(),
    # SimilarityFeature(w2v_model),
    # LexiconFeature(rusentilex_filepath, prefix_processor),
    # PatternFeature([',']),
    # EntitiesBetweenFeature(),
    # PrepositionsCountFeature(prepositions_list),
    # EntitiesFrequency(),
    # EntityAppearanceFeature(),
    # ContextPosBeforeFeature(),
    # ContextSentimentAfterFeature(rusentilex_filepath)
]

#
# Train collection
#
root = io_utils.train_root()
for n in io_utils.train_indices():
    entity_filepath = root + "art{}.ann".format(n)
    opin_filepath = root + "art{}.opin.txt".format(n)
    neutral_filepath = root + "art{}.neut.txt".format(n)
    news_filepath = root + "art{}.txt".format(n)
    vector_output = root + "art{}.vectors.txt".format(n)

    print vector_output

    entities = EntityCollection.from_file(entity_filepath)
    news = News.from_file(news_filepath, entities)
    sentiment_opins = OpinionCollection.from_file(opin_filepath)
    neutral_opins = OpinionCollection.from_file(neutral_filepath)
    filter_neutral(neutral_opins, news)

    vectors = vectorize_train(
        news,
        entities,
        [sentiment_opins, neutral_opins],
        synonyms_collection)

    vectors.save(vector_output)

#
# Test collection
#
print 'test'
root = io_utils.test_root()
for n in io_utils.test_indices():
    entity_filepath = root + "art{}.ann".format(n)
    news_filepath = root + "art{}.txt".format(n)
    vector_output = root + "art{}.vectors.txt".format(n)

    print vector_output

    entities = EntityCollection.from_file(entity_filepath)
    news = News.from_file(news_filepath, entities)

    vectors = vectorize_test(news, entities)
    vectors.save(vector_output)

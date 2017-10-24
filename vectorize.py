#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging
import numpy as np
from gensim.models.word2vec import Word2Vec

import core.environment as env

from core.source.opinion import OpinionCollection
from core.source.entity import EntityCollection
from core.source.news import News
from core.source.relations import Relation

from core.features.distance import DistanceFeature
from core.features.similarity import SimilarityFeature
from core.features.lexicon import LexiconFeature
from core.features.pattern import PatternFeature
from core.features.entities import EntitiesBetweenFeature
from core.features.prepositions import PrepositionsCountFeature

from core.processing.prefix import SentimentPrefixProcessor

import io_utils


def vectorize_train(news, entities, opinion_collections, features):
    """ Vectorize news of train collection that has opinion labeling
    """
    vectors = []
    scores = []

    sentiment_to_int = {'pos': 1, 'neg': -1, 'neu': 0}
    for opinions in opinion_collections:
        for opinion in opinions:
            # print opinion.entity_left.encode('utf-8'), opinion.entity_right.encode('utf-8')
            entities_left = entities.find_by_value(opinion.entity_left)
            entities_right = entities.find_by_value(opinion.entity_right)

            feature_vector = None
            relations_count = len(entities_left) * len(entities_right)
            for e1 in entities_left:
                for e2 in entities_right:
                    r = Relation(e1.ID, e2.ID, news)
                    r_features = np.concatenate(
                        [f.normalize(f.create(r)) for f in features])

                    if feature_vector is None:
                        feature_vector = r_features
                    else:
                        feature_vector += r_features

            if relations_count == 0:
                logging.info("- {} -> {}".format(
                    opinion.entity_left.encode('utf-8'),
                    opinion.entity_right.encode('utf-8')))
                continue

            feature_vector /= relations_count

            vectors.append(feature_vector)
            scores.append(sentiment_to_int[opinion.sentiment])

    return (vectors, scores)


def vectorize_test(news, entities, features):
    """ Vectorize news of test collection
    """
    features_by_relation = {}
    relations_count = {}
    vectors = []

    for e1 in entities:
        for e2 in entities:

            if e1.ID == e2.ID:
                continue

            s1 = news.find_sentence_by_entity(e1)
            s2 = news.find_sentence_by_entity(e2)

            # If entites from different sentences
            if not s1 == s2:
                continue

            r = Relation(e1.ID, e2.ID, news)

            # TODO: refactor env -> Environment
            relation_name = "{}->{}".format(
                env.stemmer.lemmatize_to_str(e1.value).encode('utf-8'),
                env.stemmer.lemmatize_to_str(e2.value).encode('utf-8'))

            r_features = np.concatenate(
                [f.normalize(f.create(r)) for f in features])

            features_by_relation[relation_name] = r_features

            # Increase relations count
            if relation_name not in relations_count:
                relations_count[relation_name] = 1
            else:
                relations_count[relation_name] += 1

            # add features
            if relation_name not in features_by_relation:
                features_by_relation[relation_name] = r_features
            else:
                features_by_relation[relation_name] += r_features

    for key, features in features_by_relation.iteritems():
        vectors.append(features/relations_count[key])

    return vectors


def save(vector_output, vectors, scores=None):
    with open(vector_output, 'w') as output:
        for i, v in enumerate(vectors):
            for item in v:
                output.write("%.6f " % (item))
            if scores is not None:
                output.write("{}\n".format(scores[i]))
            else:
                output.write("\n")


#
# Main
#

root = "data/Texts/"
preps_filepath = "data/prepositions.txt"
rusentilex_filepath = "data/rusentilex.csv"
w2v_model_filepath = "../tone-classifier/data/w2v/news_rusvectores2.bin.gz"

# w2v_model = Word2Vec.load_word2vec_format(w2v_model_filepath, binary=True)
prefix_processor = SentimentPrefixProcessor.from_file("data/prefixes.csv")
prepositions_list = io_utils.read_prepositions(preps_filepath)

features = [
    DistanceFeature(),
    # SimilarityFeature(w2v_model),
    # LexiconFeature(rusentilex_filepath, prefix_processor),
    # PatternFeature([',']),
    # EntitiesBetweenFeature(),
    # PrepositionsCountFeature(prepositions_list)
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

    vectors, labels = vectorize_train(
        news, entities, [sentiment_opins, neutral_opins], features)
    save(vector_output, vectors, scores=labels)

#
# Test collection
#
root = io_utils.test_root()
for n in io_utils.test_indices():
    entity_filepath = root + "art{}.ann".format(n)
    news_filepath = root + "art{}.txt".format(n)
    vector_output = root + "art{}.vectors.txt".format(n)

    print vector_output

    entities = EntityCollection.from_file(entity_filepath)
    news = News.from_file(news_filepath, entities)

    vectors = vectorize_test(news, entities, features)
    save(vector_output, vectors)

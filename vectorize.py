#!/usr/bin/python
# -*- coding: utf-8 -*-
# TODO: saves file with vectors for appropriate article.

import logging
import numpy as np
from gensim.models.word2vec import Word2Vec

from core.opinion import OpinionCollection
from core.annot import EntityCollection
from core.relations import Relation
from core.news import News

from core.features.distance import DistanceFeature
from core.features.similarity import SimilarityFeature
from core.features.lexicon import LexiconFeature
from core.processing.prefix import SentimentPrefixProcessor

root = "data/Texts/"

n = 1
annot_filepath = root + "art{}.ann".format(n)
opin_filepath = root + "art{}.opin.txt".format(n)
neutral_filepath = root + "art{}.neut.txt".format(n)
news_filepath = root + "art{}.txt".format(n)
vector_output = root + "art{}.vectors.txt".format(n)
# w2v_model_filepath = "../tone-classifier/data/w2v/news_rusvectores2.bin.gz"


# w2v_model = Word2Vec.load_word2vec_format(w2v_model_filepath, binary=True)
entities = EntityCollection.from_file(annot_filepath)
news = News.from_file(news_filepath, entities)
sentiment_opinions = OpinionCollection.from_file(opin_filepath)
# neutral_opinions = OpinionCollection.from_file(neutral_filepath)
prefix_processor = SentimentPrefixProcessor.from_file("data/prefixes.csv")

features = [
    DistanceFeature(),
    # SimilarityFeature(w2v_model),
    LexiconFeature("data/rusentilex.csv", prefix_processor)
    ]

sentiment_to_int = {'pos': 1, 'neg': -1, 'neu': 0}

vectors = []
scores = []
for opinion in sentiment_opinions:
    print opinion.entity_left.encode('utf-8'), opinion.entity_right.encode('utf-8')
    entities_left = entities.find_by_value(opinion.entity_left)
    entities_right = entities.find_by_value(opinion.entity_right)

    feature_vector = None
    relations_count = len(entities_left) * len(entities_right)
    for e1 in entities_left:
        for e2 in entities_right:
            r = Relation(e1.ID, e2.ID, news)
            r_features = np.concatenate([f.create(r) for f in features])

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

with open(vector_output, 'w') as output:
    for i, v in enumerate(vectors):
        for item in v:
            output.write("%.6f " % (item))
        output.write("{}\n".format(scores[i]))

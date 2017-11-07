#!/usr/bin/python

import io_utils
import core.environment as env
from core.source.opinion import OpinionCollection
from core.source.entity import EntityCollection
from core.source.synonyms import SynonymsCollection
from graphviz import Digraph


def create_graph(entities, opinion_collections, synonym_collection):

    def get_appropriate_entities(o):
        if synonyms_collection.has_synonym(o):
            return filter(
                lambda s: entities.has_enity_by_value(s),
                synonyms_collection.get_synonyms(o))
        elif entities.has_enity_by_value(o):
            return [o]
        else:
            return []

    dot = Digraph()

    for opinions in opinion_collections:
        for o in opinions:
            left_values = get_appropriate_entities(o.entity_left)
            right_values = get_appropriate_entities(o.entity_right)

            # TODO. We guarantee that these left and right values are not lemmatized
            if len(left_values) == 0 or len(right_values) == 0:
                print "Appropriate entity for '{}'->'{}' has not been found".format(
                    env.stemmer.lemmatize_to_str(o.entity_left).encode('utf-8'),
                    env.stemmer.lemmatize_to_str(o.entity_right).encode('utf-8'))

            # TODO: map to entities
            for entity_left in left_values:
                for entity_right in right_values:
                    el = env.stemmer.lemmatize_to_str(entity_left).encode('utf-8')
                    er = env.stemmer.lemmatize_to_str(entity_right).encode('utf-8')

                    dot.node(el)
                    dot.node(er)
                    dot.edge(el, er)

    print dot.source


synonyms_filepath = "data/synonyms.txt"
synonyms_collection = SynonymsCollection.from_file(synonyms_filepath)

root = io_utils.train_root()
for n in io_utils.train_indices():
    entity_filepath = root + "art{}.ann".format(n)
    opin_filepath = root + "art{}.opin.txt".format(n)
    vector_output = root + "art{}.vectors.txt".format(n)
    print vector_output
    entities = EntityCollection.from_file(entity_filepath)
    sentiment_opins = OpinionCollection.from_file(opin_filepath)
    vectors = create_graph(entities, [sentiment_opins], synonyms_collection)

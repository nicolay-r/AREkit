#!/usr/bin/python

import io_utils
import core.environment as env
from core.source.opinion import OpinionCollection
from core.source.entity import EntityCollection
from core.source.synonyms import SynonymsCollection
from graphviz import Digraph


def create_graph(entities, opinion_collections, synonym_collection):

    def get_appropriate_entities_value(o):
        if synonyms_collection.has_synonym(o):
            return filter(
                lambda s: entities.has_enity_by_value(s),
                synonyms_collection.get_synonyms(o))

        elif entities.has_enity_by_value(o):
            return [o]
        else:
            return []

    dot = Digraph()

    nodes = set()
    edges = 0
    for opinions in opinion_collections:
        for o in opinions:
            left_value = get_appropriate_entities_value(o.entity_left)
            right_values = get_appropriate_entities_value(o.entity_right)

            # TODO. We guarantee that these left and right values are not lemmatized
            if len(left_value) == 0 or len(right_values) == 0:
                print "Appropriate entity for '{}'->'{}' has not been found".format(
                    env.stemmer.lemmatize_to_str(o.entity_left).encode('utf-8'),
                    env.stemmer.lemmatize_to_str(o.entity_right).encode('utf-8'))
                continue

            # TODO: map to entities
            for left_v in left_value:
                for right_v in right_values:
                    # el = env.stemmer.lemmatize_to_str(entity_left).encode('utf-8')
                    # er = env.stemmer.lemmatize_to_str(entity_right).encode('utf-8')
                    assert(type(left_v) == unicode)

                    entities_left_ids = entities.get_by_value(left_v)
                    entities_right_ids = entities.get_by_value(right_v)

                    for l_id in entities_left_ids:
                        for r_id in entities_right_ids:
                            l = str(l_id)
                            r = str(r_id)
                            if (edges > 200):
                                break

                            if l not in nodes:
                                dot.node(l)

                            if r not in nodes:
                                dot.node(r)

                            dot.edge(l, r)
                            edges += 1

    print edges
    return dot


synonyms_collection = SynonymsCollection.from_file(io_utils.get_synonyms_filepath())

root = io_utils.train_root()
for n in io_utils.train_indices():
    entity_filepath = root + "art{}.ann".format(n)
    opin_filepath = root + "art{}.opin.txt".format(n)
    neut_filepath = root + "art{}.neut.txt".format(n)
    graph_output = root + "art{}.graph.jpg".format(n)
    print graph_output
    entities = EntityCollection.from_file(entity_filepath)
    sentiment_opins = OpinionCollection.from_file(opin_filepath)
    neutral_opins = OpinionCollection.from_file(neut_filepath)
    graph = create_graph(entities,
                         [sentiment_opins, neutral_opins],
                         synonyms_collection)
    graph.render(graph_output)

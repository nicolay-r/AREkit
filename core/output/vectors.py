# -*- coding: utf-8 -*-

import core.environment as env


class CommonRelationVectorCollection:

    def __init__(self):
        self.stemmer = env.stemmer
        self.vectors = {}

    def add_vector(self, vector):
        assert(isinstance(vector, CommonRelationVector))
        key = self.__create_key(vector)

        if key in self.vectors:
           print "Collection already has a key {}. Ignored".format(key.encode('utf-8'))
           return

        self.vectors[key] = vector

    def has_vector(self, vector):
        assert(isinstance(vector, CommonRelationVector))
        key = self.__create_key(vector)
        return key in self.vectors

    def __create_key(self, vector):
        key = "{}_{}".format(
            vector.value_left.encode('utf-8'),
            vector.value_right.encode('utf-8')).decode('utf-8')
        return key

    def __iter__(self):
        for v in self.vectors.itervalues():
            yield v


# TODO: maybe in core, file 'relation.py'
class CommonRelationVector:
    """ Vector of Relation between two lemmatized values of entities.
    """

    def __init__(self, entity_value_left, entity_value_right, vector, label=0):
        assert(type(entity_value_left) == unicode)
        assert(type(entity_value_right) == unicode)
        assert(type(vector) == list)
        assert(type(label) == int)

        self.value_left = env.stemmer.lemmatize_to_str(entity_value_left)
        self.value_right = env.stemmer.lemmatize_to_str(entity_value_right)
        self.vector = vector
        self.label = label

    def to_str(self):
        vector_str = ",".join(["%.6f" % v for v in self.vector])
        return "{}, {}, {}, {}".format(
            self.value_left.encode('utf-8'),
            self.value_right.encode('utf-8'),
            vector_str, self.label)

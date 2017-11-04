# -*- coding: utf-8 -*-

import io
import numpy as np
import core.environment as env
import operator


class CommonRelationVectorCollection:

    def __init__(self, vectors=None):
        self.stemmer = env.stemmer
        self.vectors = {} if vectors is None else vectors

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

    @staticmethod
    def from_file(filepath):
        """ Read the vectors from *.vectors.txt file
        """
        vectors = []
        with io.open(filepath, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                args = line.split(',')

                entity_value_left = args[0].strip()
                entity_value_right = args[1].strip()
                label = int(args[len(args) - 1])
                vector = np.array([float(args[i]) for i in range(2, len(args)-1)])

                vectors.append(CommonRelationVector(
                    entity_value_left, entity_value_right, vector, label))

        return CommonRelationVectorCollection(vectors)

    def get_by_popularity(self, limit=10):
        most_popular = sorted(self.vectors.items(), key=operator.itemgetter(1).popularity)
        return most_popular[:limit]


    def save(self, filepath):
        """ Save the vectors from *.vectors.txt file
        """
        with open(filepath, 'w') as output:
            for vector in self.vectors.itervalues():
                output.write("{}\n".format(vector.to_str()))

    def __iter__(self):
        for v in self.vectors:
            yield v


# TODO: maybe in core, file 'relation.py'
class CommonRelationVector:
    """ Vector of Relation between two lemmatized values of entities.
    """

    def __init__(self, entity_value_left, entity_value_right, vector, label=0, popularity=0):
        assert(type(entity_value_left) == unicode)
        assert(type(entity_value_right) == unicode)
        assert(isinstance(vector, np.ndarray))
        assert(type(label) == int)
        assert(type(popularity) == int)
        self.value_left = env.stemmer.lemmatize_to_str(entity_value_left)
        self.value_right = env.stemmer.lemmatize_to_str(entity_value_right)
        self.vector = vector
        self.label = label              # sentiment label or 0 in case of neutral
        self.popularity = popularity    # might be used to show an amount of Relations originally

    def set_label(self, label):
        assert(type(label) == int)
        self.label = label

    def to_str(self):
        vector_str = ",".join(["%.6f" % v for v in self.vector])
        return "{}, {}, {}, {}".format(
            self.value_left.encode('utf-8'),
            self.value_right.encode('utf-8'),
            vector_str, self.label)

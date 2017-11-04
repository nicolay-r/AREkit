import numpy as np
from math import exp

from core.source.relations import Relation


class Feature:

    def __init__():
        pass

    def calculate(self, relations):
        """ functions_list: np.min, np.max, np.sum
        """
        assert(type(relations) == list)
        results = []
        for relation in relations:
            results.append(self.create(relation))
        return self._normalize(np.concatenate((np.min(results, axis=0), np.max(results, axis=0), np.sum(results, axis=0))))

    def create(self, relation):
        """ Create feature
        """
        assert(isinstance(relation, Relation))
        raise NotImplementedError("Impelement feature create method!")

    @staticmethod
    def _normalize(vector):
        def sgn(x):
            if x > 0:
                return 1
            elif x < 0:
                return -1
            return 0

        assert(isinstance(vector, np.ndarray))
        for i in range(len(vector)):
            vector[i] = (1 - exp(-abs(vector[i]))) * sgn(vector[i])

        return vector

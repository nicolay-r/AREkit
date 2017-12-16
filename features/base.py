from math import exp

import numpy as np

from core.relations import Relation


class Base:

    def __init__():
        # TODO. pass here functions that will be used for features.
        pass

    def calculate(self, relations):
        """ functions_list: np.min, np.max, np.average
        """
        assert(type(relations) == list)
        results = []
        for relation in relations:
            results.append(self.create(relation))
        return self._normalize(
            np.concatenate((
                np.min(results, axis=0),
                np.max(results, axis=0),
                np.average(results, axis=0))))

    def feature_function_names(self):
        feature_names = self.feature_names()
        f_min = [f + '_min' for f in feature_names]
        f_max = [f + '_max' for f in feature_names]
        f_avg = [f + '_avg' for f in feature_names]
        return f_min + f_max + f_avg

    def feature_names(self):
        raise NotImplementedError("method wasn't implemented")

    def create(self, relation):
        assert(isinstance(relation, Relation))
        raise NotImplementedError("method wasn't implemented")

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

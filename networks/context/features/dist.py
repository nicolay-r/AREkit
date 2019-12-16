import numpy as np


class DistanceFeatures(object):

    @staticmethod
    def distance_feature(position, size):
        result = np.zeros(size)
        for i in xrange(len(result)):
            result[i] = i - position if i - position >= 0 else i - position + size
        return result

    @staticmethod
    def distance_abs_nearest_feature(positions, size):
        result = np.zeros(size)
        for i in xrange(len(result)):
            result[i] = min([abs(i - p) for p in positions])
        return result

import numpy as np

# TODO. Refactoring as static methods.

def distance_feature(position, size):
    result = np.zeros(size)
    for i in xrange(len(result)):
        result[i] = i - position if i - position >= 0 else i - position + size
    return result


def dist_abs_nearest_feature(positions, size):
    result = np.zeros(size)
    for i in xrange(len(result)):
        result[i] = min([abs(i - p) for p in positions])
    return result

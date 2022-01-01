import collections


class NetworkInputShapes(object):

    SYNONYMS_PER_CONTEXT = "synonyms_per_context"
    TERMS_PER_CONTEXT = "terms_per_context"
    FRAMES_PER_CONTEXT = "frames_per_context"

    def __init__(self, iter_pairs):
        assert(isinstance(iter_pairs, collections.Iterable))
        self.__d = dict()
        for key, value in iter_pairs:
            self.__d[key] = value

    def get_shape(self, key):
        return self.__d[key]

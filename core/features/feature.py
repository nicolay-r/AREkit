from math import exp


class Feature:

    def __init__():
        pass

    def create():
        """ Create feature
        """
        raise NotImplementedError("Impelement feature create method!")

    @staticmethod
    def normalize(vector):
        assert(type(vector) == list)
        return [(1 - exp(-abs(v))) * Feature.__sgn(v) for v in vector]

    @staticmethod
    def __sgn(x):
        if x > 0:
            return 1
        elif x < 0:
            return -1
        return 0

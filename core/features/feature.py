from math import exp


class Feature:

    def __init__():
        pass

    def create():
        """ Create feature
        """
        raise NotImplementedError("Impelement feature create method!")

    @staticmethod
    def _normalize(vector):
        def sgn(x):
            if x > 0:
                return 1
            elif x < 0:
                return -1
            return 0

        assert(type(vector) == list)
        return [(1 - exp(-abs(v))) * sgn(v) for v in vector]
